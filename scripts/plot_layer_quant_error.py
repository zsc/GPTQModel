#!/usr/bin/env python3
"""Plot layer-wise quantization error accumulation for a SPEC DoReFa run.

Measures two kinds of layer-indexed errors:
- Weight quantization error: relative L2 between original and quantized weights (aggregated per layer).
- Hidden-state error (accumulated along depth): relative L2 between BF16 baseline and quantized hidden states
  at each layer output.

Input:
- <run_dir>/summary.json produced by `scripts/dorefa_spec_experiments.py`

Output (default):
- <run_dir>/layer_error/layer_error_<model>.png
- <run_dir>/layer_error/layer_error_<model>.json
- <run_dir>/layer_error/index.html
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

def _load_dotenv(dotenv_path: Path) -> None:
    try:
        text = dotenv_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key:
            os.environ.setdefault(key, val)


def _maybe_load_dotenv() -> None:
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]
    for p in candidates:
        if p.exists():
            _load_dotenv(p)
            break


_maybe_load_dotenv()


DEFAULT_TEXT_PATH = os.environ.get("SPEC_TEXT_PATH", "/workspace/zhousc6@xiaopeng.com/swift_train/lgqm.txt")


@dataclass(frozen=True)
class SpecRow:
    model_key: str
    model_path: str
    config: str
    num_bits: Optional[int]
    outlier_percentile: float
    act_bits: Optional[int]
    act_outlier_percentile: float
    act_calib_chars: int
    act_calib_seq_len: int
    ppl: float
    quantile_sample_size: int
    target_layers: List[int]
    target_layer_mode: str


def _latest_run_dir(results_root: Path) -> Path:
    candidates = sorted(results_root.glob("spec_run_*"))
    if not candidates:
        raise SystemExit(f"No spec_run_* directories found under: {results_root}")
    return candidates[-1]


def _read_summary(run_dir: Path) -> List[SpecRow]:
    summary_path = run_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: List[SpecRow] = []
    for item in payload:
        target_layers_raw = item.get("target_layers") or []
        target_layers = [int(v) for v in target_layers_raw]
        target_layer_mode = str(item.get("target_layer_mode") or ("middle" if target_layers else "unknown"))
        rows.append(
            SpecRow(
                model_key=str(item["model_key"]),
                model_path=str(item["model_path"]),
                config=str(item.get("config", "")),
                num_bits=item.get("num_bits", None),
                outlier_percentile=float(item.get("outlier_percentile", 1.0)),
                act_bits=item.get("act_bits", None),
                act_outlier_percentile=float(item.get("act_outlier_percentile", 1.0)),
                act_calib_chars=int(item.get("act_calib_chars", 20000)),
                act_calib_seq_len=int(item.get("act_calib_seq_len", 256)),
                ppl=float(item.get("ppl", float("nan"))),
                quantile_sample_size=int(item.get("quantile_sample_size", 0)),
                target_layers=target_layers,
                target_layer_mode=target_layer_mode,
            )
        )
    return rows


def _infer_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise AttributeError("Unsupported model layout: cannot find transformer layers")


def _iter_layer_weights(
    model,
    target_layer_indices: Iterable[int],
) -> Iterable[Tuple[int, str, torch.Tensor]]:
    layers = _infer_layers(model)
    for layer_idx in target_layer_indices:
        layer = layers[layer_idx]

        # Qwen/MiniCPM/Gemma-style
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    yield layer_idx, f"layer{layer_idx}.self_attn.{proj_name}.weight", proj.weight

        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(mlp, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    yield layer_idx, f"layer{layer_idx}.mlp.{proj_name}.weight", proj.weight

        # InternLM2-style
        attn2 = getattr(layer, "attention", None)
        if attn2 is not None:
            for proj_name in ("wqkv", "wo"):
                proj = getattr(attn2, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    yield layer_idx, f"layer{layer_idx}.attention.{proj_name}.weight", proj.weight

        ff = getattr(layer, "feed_forward", None)
        if ff is not None:
            for proj_name in ("w1", "w2", "w3"):
                proj = getattr(ff, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    yield layer_idx, f"layer{layer_idx}.feed_forward.{proj_name}.weight", proj.weight


def _compute_quantile_bounds(
    weight: torch.Tensor,
    outlier_percentile: float,
    *,
    sample_size: int,
) -> Tuple[float, float]:
    if float(outlier_percentile) <= 0.0:
        wf = weight.detach()
        return float(wf.min().item()), float(wf.max().item())
    q = outlier_percentile / 100.0
    wf = weight.detach().float().reshape(-1)
    if sample_size > 0 and wf.numel() > sample_size:
        step = max(1, wf.numel() // sample_size)
        wf = wf[::step][:sample_size]
    sample = wf.detach().cpu().numpy()
    w_min = float(np.quantile(sample, q))
    w_max = float(np.quantile(sample, 1.0 - q))
    return w_min, w_max


def _load_or_build_bounds_cache(
    *,
    cache_path: Path,
    model,
    target_layers: List[int],
    outlier_percentile: float,
    sample_size: int,
) -> Dict[str, Tuple[float, float]]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return {k: (float(v[0]), float(v[1])) for k, v in payload.items()}

    bounds: Dict[str, Tuple[float, float]] = {}
    for _layer_idx, name, weight in _iter_layer_weights(model, target_layers):
        bounds[name] = _compute_quantile_bounds(weight, outlier_percentile, sample_size=sample_size)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(bounds, ensure_ascii=False, indent=2), encoding="utf-8")
    return bounds


def _dorefa_quantize_tensor(wf: torch.Tensor, *, num_bits: int, bounds: Tuple[float, float]) -> torch.Tensor:
    qmin = -(2 ** (num_bits - 1) - 1)
    qmax = 2 ** (num_bits - 1) - 1

    w_min, w_max = bounds
    outlier_mask = (wf < w_min) | (wf > w_max)

    scale = (w_max / qmax) if w_max > 0 else 1.0
    wq = torch.clamp(torch.round(wf / scale), qmin, qmax) * scale
    return torch.where(outlier_mask, wf, wq)


def _compute_tensor_quantile_bounds(
    tensor: torch.Tensor,
    outlier_percentile: float,
    *,
    sample_size: int,
) -> Tuple[float, float]:
    if float(outlier_percentile) <= 0.0:
        tf = tensor.detach()
        return float(tf.min().item()), float(tf.max().item())
    q = outlier_percentile / 100.0
    tf = tensor.detach().float().reshape(-1)
    if sample_size > 0 and tf.numel() > sample_size:
        step = max(1, tf.numel() // sample_size)
        tf = tf[::step][:sample_size]
    sample = tf.detach().cpu().numpy()
    t_min = float(np.quantile(sample, q))
    t_max = float(np.quantile(sample, 1.0 - q))
    return float(t_min), float(t_max)


def _load_or_build_activation_bounds_cache(
    *,
    cache_path: Path,
    model,
    tokenizer,
    calib_text: str,
    target_layers: List[int],
    outlier_percentile: float,
    seq_len: int,
    sample_size: int,
) -> Dict[int, Tuple[float, float]]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        bounds_raw = payload.get("bounds", payload)
        return {int(k): (float(v[0]), float(v[1])) for k, v in bounds_raw.items()}

    inputs = tokenizer(calib_text, return_tensors="pt", truncation=True, max_length=seq_len)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden_states = list(getattr(out, "hidden_states", None) or [])
    if not hidden_states:
        inner = getattr(model, "model", None)
        if inner is not None and inner is not model:
            try:
                inner_out = inner(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask", None),
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = list(getattr(inner_out, "hidden_states", None) or [])
            except Exception:
                hidden_states = []
    if not hidden_states:
        raise RuntimeError("Model did not return hidden_states; cannot calibrate activation quantization.")

    bounds: Dict[int, Tuple[float, float]] = {}
    for layer_idx in target_layers:
        hs = hidden_states[layer_idx + 1]
        bounds[layer_idx] = _compute_tensor_quantile_bounds(
            hs, outlier_percentile, sample_size=sample_size
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "seq_len": int(seq_len),
        "outlier_percentile": float(outlier_percentile),
        "sample_size": int(sample_size),
        "bounds": {str(k): [float(v[0]), float(v[1])] for k, v in bounds.items()},
    }
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return bounds


def _register_activation_quant_hooks(
    model,
    *,
    target_layers: List[int],
    bounds_by_layer: Dict[int, Tuple[float, float]],
    num_bits: int,
) -> List[Any]:
    layers = _infer_layers(model)
    handles: List[Any] = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            bounds = bounds_by_layer.get(layer_idx)
            if bounds is None:
                return output
            if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                hs = output[0]
                hs_q = _dorefa_quantize_tensor(hs.detach().float(), num_bits=num_bits, bounds=bounds).to(
                    dtype=hs.dtype
                )
                return (hs_q,) + output[1:]
            if torch.is_tensor(output):
                return _dorefa_quantize_tensor(output.detach().float(), num_bits=num_bits, bounds=bounds).to(
                    dtype=output.dtype
                )
            return output

        return hook

    for layer_idx in target_layers:
        handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    return handles


def _apply_quant_and_measure_weight_error(
    model,
    *,
    num_bits: int,
    target_layers: List[int],
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, List[float]]:
    num_layers = len(_infer_layers(model))
    sum_sq_err = [0.0] * num_layers
    sum_sq_orig = [0.0] * num_layers
    count = [0] * num_layers

    for layer_idx, name, weight in _iter_layer_weights(model, target_layers):
        wf = weight.detach().float()
        wq = _dorefa_quantize_tensor(wf, num_bits=num_bits, bounds=bounds[name])

        diff = wq - wf
        sum_sq_err[layer_idx] += float(diff.pow(2).sum().item())
        sum_sq_orig[layer_idx] += float(wf.pow(2).sum().item())
        count[layer_idx] += int(wf.numel())

        weight.data.copy_(wq.to(dtype=weight.dtype))

    rel_l2: List[float] = []
    mse: List[float] = []
    for i in range(num_layers):
        if count[i] == 0 or sum_sq_orig[i] == 0.0:
            rel_l2.append(0.0)
            mse.append(0.0)
            continue
        rel_l2.append(float(math.sqrt(sum_sq_err[i] / sum_sq_orig[i])))
        mse.append(float(sum_sq_err[i] / count[i]))

    return {"weight_rel_l2": rel_l2, "weight_mse": mse}


@torch.inference_mode()
def _hidden_states_for_text(model, tokenizer, text: str, *, seq_len: int) -> List[torch.Tensor]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden_states = list(getattr(out, "hidden_states", None) or [])
    if not hidden_states:
        # Some remote-code models do not plumb `output_hidden_states=True` through their
        # top-level forward() (e.g. wrappers around an inner HF model). Try the inner
        # `model.model` as a fallback.
        inner = getattr(model, "model", None)
        if inner is not None and inner is not model:
            try:
                inner_out = inner(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask", None),
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = list(getattr(inner_out, "hidden_states", None) or [])
            except Exception:
                hidden_states = []
    if not hidden_states:
        raise RuntimeError("Model did not return hidden_states; cannot compute activation error.")
    return hidden_states


def _activation_error_by_layer(
    baseline_hidden_states: List[torch.Tensor],
    quant_hidden_states: List[torch.Tensor],
) -> List[float]:
    if len(baseline_hidden_states) != len(quant_hidden_states):
        raise ValueError("Hidden states length mismatch between baseline and quantized runs.")

    num_layers = len(baseline_hidden_states) - 1
    eps = 1e-12
    errors: List[float] = []

    for layer_idx in range(num_layers):
        b = baseline_hidden_states[layer_idx + 1].float()
        q = quant_hidden_states[layer_idx + 1].float()
        diff = q - b
        num = torch.linalg.norm(diff)
        den = torch.linalg.norm(b) + eps
        errors.append(float((num / den).item()))

    return errors


def _load_model_and_tokenizer(model_path: str, *, use_fast: Optional[bool]) -> Tuple[Any, Any]:
    tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if use_fast is not None:
        tokenizer_kwargs["use_fast"] = use_fast
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", **model_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.to("cuda")

    model.eval()
    llm = getattr(model, "llm", None)
    if llm is not None and hasattr(llm, "generate"):
        if getattr(llm.config, "pad_token_id", None) is None:
            llm.config.pad_token_id = tokenizer.pad_token_id
        llm.eval()
        model = llm
    return model, tokenizer


def _use_fast_tokenizer_for_model(model_key: str) -> Optional[bool]:
    return False if model_key == "qwen3" else None


def _load_text(text_path: str, max_chars: int) -> str:
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text[:max_chars] if len(text) > max_chars else text


def _plot_model(
    *,
    model_key: str,
    plot_indices: List[int],
    configs: List[Dict[str, Any]],
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    x = plot_indices
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax_act, ax_w = axes
    ax_act.set_title(f"{model_key}: Hidden-State Error Accumulation (rel L2)")
    ax_act.set_ylabel("rel L2 (log)")
    ax_act.set_yscale("log")
    ax_act.grid(True, alpha=0.25)

    ax_w.set_title("Weight Quantization Error (per layer, rel L2)")
    ax_w.set_xlabel("Layer index")
    ax_w.set_ylabel("rel L2 (log)")
    ax_w.set_yscale("log")
    ax_w.grid(True, alpha=0.25)

    for cfg in configs:
        label = cfg["label"]
        act = [max(1e-12, float(v)) for v in cfg["activation_rel_l2_by_layer"]]
        w = [max(1e-12, float(v)) for v in cfg["weight_rel_l2_by_layer"]]
        ax_act.plot(x, act, label=label, linewidth=1.8)
        ax_w.plot(x, w, label=label, linewidth=1.8)

    ax_act.legend(loc="upper left", fontsize=9, ncol=2, frameon=False)
    ax_w.legend(loc="upper left", fontsize=9, ncol=2, frameon=False)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _write_index_html(out_dir: Path, models: List[Dict[str, str]]) -> None:
    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8' />")
    parts.append("<meta name='viewport' content='width=device-width,initial-scale=1' />")
    parts.append("<title>Layer Error Report</title>")
    parts.append(
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;background:#0b0f17;color:#e7ecf5}"
        ".card{border:1px solid rgba(231,236,245,.12);border-radius:14px;padding:16px;margin-bottom:16px;background:rgba(15,22,35,.85)}"
        "img{max-width:100%;border-radius:12px;border:1px solid rgba(231,236,245,.12)}"
        "a{color:#7aa2ff;text-decoration:none}a:hover{text-decoration:underline}"
        "code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;"
        "background:rgba(231,236,245,.08);border:1px solid rgba(231,236,245,.12);padding:1px 6px;border-radius:8px}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append("<h1>Layer-wise Quantization Error Accumulation</h1>")
    for m in models:
        parts.append("<div class='card'>")
        parts.append(f"<h2>{m['model_key']}</h2>")
        parts.append(
            f"<div>JSON: <a href='{m['json_name']}'><code>{m['json_name']}</code></a></div>"
        )
        parts.append(
            f"<div style='margin-top:10px'><img src='{m['png_name']}' alt='{m['model_key']}' /></div>"
        )
        parts.append("</div>")
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--text-path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--text-chars", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument(
        "--include-ends",
        action="store_true",
        help="Include first/last transformer blocks in plots (default: exclude).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys to analyze (default: all in summary.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/layer_error)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or _latest_run_dir(args.results_root)
    rows = _read_summary(run_dir)

    selected_models = None
    if args.models:
        selected_models = {m.strip() for m in args.models.split(",") if m.strip()}

    by_model: Dict[str, List[SpecRow]] = {}
    for r in rows:
        if selected_models is not None and r.model_key not in selected_models:
            continue
        by_model.setdefault(r.model_key, []).append(r)

    out_dir = args.out_dir or (run_dir / "layer_error")
    out_dir.mkdir(parents=True, exist_ok=True)

    text = _load_text(args.text_path, args.text_chars)

    index_models: List[Dict[str, str]] = []

    for model_key in sorted(by_model.keys()):
        model_rows = by_model[model_key]
        model_path = model_rows[0].model_path
        use_fast = _use_fast_tokenizer_for_model(model_key)

        cfg_keys = {
            (
                r.num_bits,
                r.outlier_percentile,
                r.act_bits,
                r.act_outlier_percentile,
                r.act_calib_chars,
                r.act_calib_seq_len,
                str(r.target_layer_mode or "middle"),
                tuple(int(v) for v in (r.target_layers or [])),
                r.config,
            )
            for r in model_rows
            if r.num_bits is not None
        }
        quant_cfgs = sorted(
            [
                {
                    "num_bits": int(bits or 0),
                    "outlier_percentile": float(w_outlier),
                    "act_bits": (None if act_bits is None else int(act_bits)),
                    "act_outlier_percentile": float(a_outlier),
                    "act_calib_chars": int(a_chars),
                    "act_calib_seq_len": int(a_seq),
                    "target_layer_mode": str(layer_mode),
                    "target_layers": list(target_layers),
                    "config_label": str(cfg_label),
                }
                for (
                    bits,
                    w_outlier,
                    act_bits,
                    a_outlier,
                    a_chars,
                    a_seq,
                    layer_mode,
                    target_layers,
                    cfg_label,
                ) in cfg_keys
            ],
            key=lambda d: (
                int(d["num_bits"]),
                float(d["outlier_percentile"]),
                str(d.get("target_layer_mode", "middle")),
                0 if d["act_bits"] is None else 1,
                0 if d["act_bits"] is None else int(d["act_bits"]),
                str(d["config_label"]),
            ),
        )
        sample_size = max((r.quantile_sample_size for r in model_rows), default=200_000)

        print("=" * 90)
        print(f"[Analyze] {model_key} | {model_path}")
        print(
            "  configs: "
            + ", ".join(
                [
                    f"{c['config_label']} (W={c['num_bits']}bit,{c['outlier_percentile']:g}%; A={'-' if c['act_bits'] is None else str(c['act_bits'])+'bit'} )"
                    for c in quant_cfgs
                ]
            )
        )
        print(f"  seq_len={args.seq_len} | text_chars={args.text_chars} | quantile_sample_size={sample_size}")

        # Baseline hidden states
        base_model, base_tok = _load_model_and_tokenizer(model_path, use_fast=use_fast)
        layers = _infer_layers(base_model)
        num_layers = len(layers)
        if args.include_ends:
            plot_indices = list(range(0, num_layers))
        else:
            plot_indices = list(range(1, max(1, num_layers - 1)))
        baseline_hidden_states = _hidden_states_for_text(base_model, base_tok, text, seq_len=args.seq_len)

        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_results: List[Dict[str, Any]] = []

        for cfg in quant_cfgs:
            num_bits = int(cfg["num_bits"])
            outlier_percentile = float(cfg["outlier_percentile"])
            act_bits = cfg["act_bits"]
            act_outlier_percentile = float(cfg["act_outlier_percentile"])
            act_calib_chars = int(cfg["act_calib_chars"])
            act_calib_seq_len = int(cfg["act_calib_seq_len"])
            target_layers = [int(v) for v in (cfg.get("target_layers") or [])]
            if not target_layers:
                target_layers = list(range(1, num_layers - 1))
            target_layer_mode = str(cfg.get("target_layer_mode") or "middle")
            scope_suffix = "_allblocks" if target_layer_mode == "all" else ""

            model, tok = _load_model_and_tokenizer(model_path, use_fast=use_fast)
            bounds_cache = run_dir / "quantile_cache" / f"{model_key}_outlier{outlier_percentile:g}{scope_suffix}.json"
            if not bounds_cache.exists():
                bounds_cache = run_dir / "quantile_cache" / f"{model_key}_outlier{outlier_percentile:g}.json"
            bounds = _load_or_build_bounds_cache(
                cache_path=bounds_cache,
                model=model,
                target_layers=target_layers,
                outlier_percentile=float(outlier_percentile),
                sample_size=int(sample_size),
            )

            weight_err = _apply_quant_and_measure_weight_error(
                model,
                num_bits=int(num_bits),
                target_layers=target_layers,
                bounds=bounds,
            )

            hooks: List[Any] = []
            if act_bits is not None:
                calib_text = _load_text(args.text_path, act_calib_chars)
                act_cache = (
                    run_dir
                    / "activation_cache"
                    / f"{model_key}_W{num_bits}_outlier{outlier_percentile:g}_A{int(act_bits)}_outlier{act_outlier_percentile:g}_seqlen{act_calib_seq_len}{scope_suffix}.json"
                )
                if not act_cache.exists():
                    act_cache = (
                        run_dir
                        / "activation_cache"
                        / f"{model_key}_W{num_bits}_outlier{outlier_percentile:g}_A{int(act_bits)}_outlier{act_outlier_percentile:g}_seqlen{act_calib_seq_len}.json"
                    )
                act_bounds = _load_or_build_activation_bounds_cache(
                    cache_path=act_cache,
                    model=model,
                    tokenizer=tok,
                    calib_text=calib_text,
                    target_layers=target_layers,
                    outlier_percentile=act_outlier_percentile,
                    seq_len=act_calib_seq_len,
                    sample_size=int(sample_size),
                )
                hooks = _register_activation_quant_hooks(
                    model,
                    target_layers=target_layers,
                    bounds_by_layer=act_bounds,
                    num_bits=int(act_bits),
                )

            quant_hidden_states = _hidden_states_for_text(model, tok, text, seq_len=args.seq_len)
            for h in hooks:
                h.remove()
            act_err_full = _activation_error_by_layer(baseline_hidden_states, quant_hidden_states)
            act_err = [act_err_full[i] for i in plot_indices]
            w_rel_l2_full = weight_err["weight_rel_l2"]
            w_mse_full = weight_err["weight_mse"]
            w_rel_l2 = [w_rel_l2_full[i] for i in plot_indices]
            w_mse = [w_mse_full[i] for i in plot_indices]

            label = str(cfg.get("config_label") or f"{int(num_bits)}-bit ({outlier_percentile:g}%)")
            model_results.append(
                {
                    "label": label,
                    "num_bits": int(num_bits),
                    "outlier_percentile": float(outlier_percentile),
                    "act_bits": act_bits,
                    "act_outlier_percentile": act_outlier_percentile,
                    "act_calib_chars": act_calib_chars,
                    "act_calib_seq_len": act_calib_seq_len,
                    "target_layer_mode": target_layer_mode,
                    "target_layers": target_layers,
                    "activation_rel_l2_by_layer": act_err,
                    "weight_rel_l2_by_layer": w_rel_l2,
                    "weight_mse_by_layer": w_mse,
                    "plot_layer_indices": plot_indices,
                    "activation_rel_l2_by_layer_full": act_err_full,
                    "weight_rel_l2_by_layer_full": w_rel_l2_full,
                    "weight_mse_by_layer_full": w_mse_full,
                }
            )

            del model
            del quant_hidden_states
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out_json = out_dir / f"layer_error_{model_key}.json"
        out_png = out_dir / f"layer_error_{model_key}.png"

        out_payload = {
            "model_key": model_key,
            "model_path": model_path,
            "num_layers": num_layers,
            "plot_layer_indices": plot_indices,
            "seq_len": args.seq_len,
            "text_path": args.text_path,
            "text_chars": args.text_chars,
            "quantile_sample_size": sample_size,
            "configs": model_results,
        }
        out_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _plot_model(model_key=model_key, plot_indices=plot_indices, configs=model_results, out_png=out_png)

        index_models.append({"model_key": model_key, "json_name": out_json.name, "png_name": out_png.name})

        del baseline_hidden_states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_index_html(out_dir, index_models)
    print(f"\n✓ Wrote: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
