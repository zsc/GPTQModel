#!/usr/bin/env python3
"""
Run the DoReFa weight-only experiments described in SPEC.md.

Notes:
- This is NOT GPTQ/AWQ packing; it keeps weights in BF16 but restricts them to
  discrete levels (per-tensor symmetric-ish quantization).
- Outliers are preserved (left unquantized) based on per-tensor quantiles.
- Optional: also quantize inter-layer activations via forward hooks (W+A).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def _load_dotenv(dotenv_path: Path) -> None:
    """Minimal .env loader (avoid extra dependency).

    Supports lines like `KEY=VALUE` and ignores blank lines / comments.
    Existing environment variables are not overwritten.
    """

    try:
        text = dotenv_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        os.environ.setdefault(key, val)


def _maybe_load_dotenv() -> None:
    # Prefer CWD, but also try repo root (scripts/..).
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]
    for p in candidates:
        if p.exists():
            _load_dotenv(p)
            break


_maybe_load_dotenv()


DEFAULT_TEXT_PATH = os.environ.get("SPEC_TEXT_PATH", "/workspace/zhousc6@xiaopeng.com/swift_train/lgqm.txt")


@dataclass(frozen=True)
class ExperimentConfig:
    label: str
    bits: Optional[int]
    outlier_percentile: float = 1.0
    act_bits: Optional[int] = None
    act_outlier_percentile: float = 1.0

    def with_prefix(self, prefix: str) -> "ExperimentConfig":
        if not prefix:
            return self
        return ExperimentConfig(
            label=f"{prefix}{self.label}",
            bits=self.bits,
            outlier_percentile=self.outlier_percentile,
            act_bits=self.act_bits,
            act_outlier_percentile=self.act_outlier_percentile,
        )


def _resolve_internlm_snapshot_path(model_path: str) -> str:
    path = Path(model_path)
    if (path / "config.json").exists():
        return str(path)
    snapshots = sorted((path / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"Could not resolve InternLM snapshot path from: {model_path}")
    return str(snapshots[-1])


def _load_text(text_path: str, max_chars: int) -> str:
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text[:max_chars] if len(text) > max_chars else text


def _infer_layers(model) -> List[torch.nn.Module]:
    # Common HF causal LM layouts.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise AttributeError("Unsupported model layout: cannot find transformer layers")


def _iter_target_weights(
    model,
    target_layer_indices: Iterable[int],
) -> Iterable[Tuple[str, torch.Tensor]]:
    layers = _infer_layers(model)
    for layer_idx in target_layer_indices:
        layer = layers[layer_idx]

        # Attention projections (Qwen/MiniCPM/Gemma-style)
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue
                yield f"layer{layer_idx}.self_attn.{proj_name}.weight", proj.weight

        # MLP projections (Qwen/MiniCPM/Gemma-style)
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(mlp, proj_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue
                yield f"layer{layer_idx}.mlp.{proj_name}.weight", proj.weight

            # Qwen3-MoE style experts: mlp.experts[i].{gate,up,down}_proj
            experts = getattr(mlp, "experts", None)
            if experts is not None:
                for expert_idx, expert in enumerate(experts):
                    for proj_name in ("gate_proj", "up_proj", "down_proj"):
                        proj = getattr(expert, proj_name, None)
                        if proj is None or not hasattr(proj, "weight"):
                            continue
                        yield (
                            f"layer{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight",
                            proj.weight,
                        )

            shared_expert = getattr(mlp, "shared_expert", None)
            if shared_expert is not None:
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    proj = getattr(shared_expert, proj_name, None)
                    if proj is None or not hasattr(proj, "weight"):
                        continue
                    yield f"layer{layer_idx}.mlp.shared_expert.{proj_name}.weight", proj.weight

        # InternLM2-style attention/feed-forward.
        attn = getattr(layer, "attention", None)
        if attn is not None:
            for proj_name in ("wqkv", "wo"):
                proj = getattr(attn, proj_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue
                yield f"layer{layer_idx}.attention.{proj_name}.weight", proj.weight

        ff = getattr(layer, "feed_forward", None)
        if ff is not None:
            for proj_name in ("w1", "w2", "w3"):
                proj = getattr(ff, proj_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue
                yield f"layer{layer_idx}.feed_forward.{proj_name}.weight", proj.weight


def _compute_quantile_bounds(
    weight: torch.Tensor,
    outlier_percentile: float,
    *,
    sample_size: int,
) -> Tuple[float, float]:
    if float(outlier_percentile) <= 0.0:
        wf = weight.detach()
        return float(wf.min().item()), float(wf.max().item())
    # Keep (outlier_percentile)% on each tail unquantized, matching SPEC.md pseudocode.
    q = outlier_percentile / 100.0
    wf = weight.detach().float().reshape(-1)
    if sample_size > 0 and wf.numel() > sample_size:
        # Deterministic strided sampling avoids expensive full-tensor quantiles.
        step = max(1, wf.numel() // sample_size)
        wf = wf[::step][:sample_size]

    # Use CPU quantiles to avoid GPU-side full sorts.
    sample = wf.detach().cpu().numpy()
    w_min = float(np.quantile(sample, q))
    w_max = float(np.quantile(sample, 1.0 - q))
    return float(w_min), float(w_max)


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


def _load_or_build_bounds_cache(
    *,
    cache_path: Path,
    model,
    target_layer_indices: List[int],
    outlier_percentile: float,
    sample_size: int,
) -> Dict[str, Tuple[float, float]]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return {k: (float(v[0]), float(v[1])) for k, v in payload.items()}

    bounds: Dict[str, Tuple[float, float]] = {}
    for name, weight in _iter_target_weights(model, target_layer_indices):
        bounds[name] = _compute_quantile_bounds(weight, outlier_percentile, sample_size=sample_size)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(bounds, ensure_ascii=False, indent=2), encoding="utf-8")
    return bounds


def _dorefa_quantize_inplace(
    weight: torch.Tensor,
    *,
    num_bits: int,
    bounds: Tuple[float, float],
) -> None:
    # SPEC.md uses a symmetric integer range but scales using w_max only.
    qmin = -(2 ** (num_bits - 1) - 1)
    qmax = 2 ** (num_bits - 1) - 1

    w_min, w_max = bounds
    wf = weight.detach().float()
    outlier_mask = (wf < w_min) | (wf > w_max)

    scale = (w_max / qmax) if w_max > 0 else 1.0
    wq = torch.clamp(torch.round(wf / scale), qmin, qmax) * scale
    wq = wq.to(dtype=weight.dtype)

    weight.data.copy_(torch.where(outlier_mask, weight.data, wq))


def _dorefa_quantize_tensor(
    tensor: torch.Tensor,
    *,
    num_bits: int,
    bounds: Tuple[float, float],
) -> torch.Tensor:
    qmin = -(2 ** (num_bits - 1) - 1)
    qmax = 2 ** (num_bits - 1) - 1

    t_min, t_max = bounds
    tf = tensor.detach().float()
    outlier_mask = (tf < t_min) | (tf > t_max)

    scale = (t_max / qmax) if t_max > 0 else 1.0
    tq = torch.clamp(torch.round(tf / scale), qmin, qmax) * scale
    tq = tq.to(dtype=tensor.dtype)
    return torch.where(outlier_mask, tensor, tq)


def _load_or_build_activation_bounds_cache(
    *,
    cache_path: Path,
    model,
    tokenizer,
    calib_text: str,
    target_layer_indices: List[int],
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
        # Some remote-code models ignore `output_hidden_states=True` at the top level
        # but wrap a regular HF text model in `model.model`.
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
    for layer_idx in target_layer_indices:
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
    target_layer_indices: List[int],
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
                hs_q = _dorefa_quantize_tensor(hs, num_bits=num_bits, bounds=bounds)
                return (hs_q,) + output[1:]
            if torch.is_tensor(output):
                return _dorefa_quantize_tensor(output, num_bits=num_bits, bounds=bounds)
            return output

        return hook

    for layer_idx in target_layer_indices:
        handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    return handles


@torch.inference_mode()
def _calculate_ppl(
    model,
    tokenizer,
    text: str,
    *,
    n_ctx: int = 2048,
    stride: Optional[int] = None,
) -> float:
    stride = stride or (n_ctx // 2)
    encodings = tokenizer(text, truncation=False, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    attention_mask = getattr(encodings, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    seq_len = input_ids.size(1)
    total_nll = 0.0
    total_tokens = 0
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + n_ctx, seq_len)
        trg_len = end_loc - prev_end_loc

        input_chunk = input_ids[:, begin_loc:end_loc]
        if attention_mask is not None:
            attn_chunk = attention_mask[:, begin_loc:end_loc]
        else:
            attn_chunk = torch.ones_like(input_chunk)
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_chunk, attention_mask=attn_chunk, labels=target_ids, use_cache=False)
        loss = getattr(outputs, "loss", None)

        if loss is not None:
            total_nll += (loss.float().item() * trg_len)
            total_tokens += trg_len
        else:
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError("Model output has neither `.loss` nor `.logits`; cannot compute PPL.")
            logits = logits.float()
            # Match HF causal LM loss: shift logits/labels by one.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
            nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            valid = int((shift_labels != -100).sum().item())
            total_nll += float(nll.item())
            total_tokens += valid

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if total_tokens == 0:
        return float("inf")
    return float(math.exp(total_nll / total_tokens))


@torch.inference_mode()
def _generate_samples(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> List[Dict[str, str]]:
    # MiniCPM-o is a multimodal/chat-style checkpoint. With raw free-form prompts,
    # it tends to degenerate (e.g., repeating punctuation). Wrap prompts as a
    # "continue this text" user message to get meaningful continuations.
    tokenizer_name = tokenizer.__class__.__name__.lower()
    use_chat_continue = "minicpmo" in tokenizer_name

    samples: List[Dict[str, str]] = []
    for prompt in prompts:
        if use_chat_continue and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": f"请续写以下文本：\n{prompt}"}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = prompt

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        )
        # Decode only newly generated tokens (avoid brittle string slicing).
        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        continuation = tokenizer.decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        samples.append({"prompt": prompt, "continuation": continuation})
    return samples


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap_text_model(model: Any) -> Any:
    """Return the text-only LLM module if `model` is a multimodal wrapper.

    Some multimodal checkpoints (e.g. MiniCPM-o) wrap the causal LM in `model.llm`
    and override `.generate()` to require vision inputs. For SPEC text-only eval,
    we want to run/quantize the underlying LLM.
    """

    llm = getattr(model, "llm", None)
    if llm is not None and hasattr(llm, "generate"):
        return llm
    return model


def _load_model_and_tokenizer(
    model_path: str,
    *,
    use_fast: Optional[bool],
    torch_dtype: Optional[Any] = None,
    device_map: Optional[str] = "auto",
) -> Tuple[Any, Any]:
    tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if use_fast is not None:
        tokenizer_kwargs["use_fast"] = use_fast
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model_kwargs.setdefault("low_cpu_mem_usage", True)

    # Some environments set WORLD_SIZE=1 even when not launched via torchrun, which
    # makes Transformers interpret device_map="auto" as TP and require LOCAL_RANK.
    if (
        model_kwargs.get("device_map") == "auto"
        and int(os.environ.get("WORLD_SIZE", "0") or "0") > 0
        and not os.environ.get("LOCAL_RANK")
    ):
        model_kwargs["device_map"] = "cuda" if torch.cuda.is_available() else None

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if torch.cuda.is_available():
            model.to("cuda")

    model.eval()
    text_model = _unwrap_text_model(model)
    if text_model is not model:
        # Keep tokenizer/config aligned after unwrapping.
        if getattr(text_model.config, "pad_token_id", None) is None:
            text_model.config.pad_token_id = tokenizer.pad_token_id
        text_model.eval()
        model = text_model
    # Ensure the model is on GPU when available; FP8 checkpoints can default to CPU.
    if torch.cuda.is_available():
        try:
            param_device = next(model.parameters()).device
        except StopIteration:
            param_device = torch.device("cpu")
        if param_device.type == "cpu":
            model.to("cuda")
    return model, tokenizer


def _run_single(
    *,
    model_key: str,
    model_path: str,
    use_fast_tokenizer: Optional[bool],
    text_path: str,
    max_chars: int,
    results_dir: Path,
    prompts: List[str],
    n_ctx: int,
    stride: int,
    config: ExperimentConfig,
    quantize_all_blocks: bool,
    seed: int,
    quantile_sample_size: int,
    act_calib_chars: int,
    act_calib_seq_len: int,
    torch_dtype: Optional[Any] = None,
    device_map: Optional[str] = "auto",
    compute_ppl: bool = True,
    existing_ppl: Optional[float] = None,
) -> Dict[str, Any]:
    _seed_all(seed)
    model, tokenizer = _load_model_and_tokenizer(
        model_path,
        use_fast=use_fast_tokenizer,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    layers = _infer_layers(model)
    if quantize_all_blocks:
        target_layers = list(range(0, len(layers)))
        target_layer_mode = "all"
    else:
        target_layers = list(range(1, len(layers) - 1))
        target_layer_mode = "middle"

    if config.bits is not None:
        scope_suffix = "_allblocks" if quantize_all_blocks else ""
        bounds_cache_path = (
            results_dir
            / "quantile_cache"
            / f"{model_key}_outlier{config.outlier_percentile:g}{scope_suffix}.json"
        )

        bounds = _load_or_build_bounds_cache(
            cache_path=bounds_cache_path,
            model=model,
            target_layer_indices=target_layers,
            outlier_percentile=config.outlier_percentile,
            sample_size=quantile_sample_size,
        )
        for name, weight in _iter_target_weights(model, target_layers):
            _dorefa_quantize_inplace(weight, num_bits=config.bits, bounds=bounds[name])

    hooks: List[Any] = []
    if config.act_bits is not None:
        calib_text = _load_text(text_path, max_chars=act_calib_chars)
        scope_suffix = "_allblocks" if quantize_all_blocks else ""
        act_cache_path = (
            results_dir
            / "activation_cache"
            / f"{model_key}_W{config.bits}_outlier{config.outlier_percentile:g}_A{config.act_bits}_outlier{config.act_outlier_percentile:g}_seqlen{act_calib_seq_len}{scope_suffix}.json"
        )
        act_bounds = _load_or_build_activation_bounds_cache(
            cache_path=act_cache_path,
            model=model,
            tokenizer=tokenizer,
            calib_text=calib_text,
            target_layer_indices=target_layers,
            outlier_percentile=config.act_outlier_percentile,
            seq_len=act_calib_seq_len,
            sample_size=quantile_sample_size,
        )
        hooks = _register_activation_quant_hooks(
            model,
            target_layer_indices=target_layers,
            bounds_by_layer=act_bounds,
            num_bits=int(config.act_bits),
        )

    if compute_ppl:
        text = _load_text(text_path, max_chars=max_chars)
        ppl = _calculate_ppl(model, tokenizer, text, n_ctx=n_ctx, stride=stride)
    else:
        ppl = float(existing_ppl) if existing_ppl is not None else float("nan")
    samples = _generate_samples(model, tokenizer, prompts)
    for h in hooks:
        h.remove()

    payload: Dict[str, Any] = {
        "model_key": model_key,
        "model_path": model_path,
        "config": config.label,
        "num_bits": config.bits,
        "outlier_percentile": config.outlier_percentile,
        "act_bits": config.act_bits,
        "act_outlier_percentile": config.act_outlier_percentile,
        "act_calib_chars": act_calib_chars,
        "act_calib_seq_len": act_calib_seq_len,
        "quantile_sample_size": quantile_sample_size,
        "target_layers": target_layers,
        "target_layer_mode": target_layer_mode,
        "ppl": round(float(ppl), 6),
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "prompts": prompts,
        "samples": samples,
    }

    # Cleanup aggressively between runs.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write JSON results and caches.",
    )
    parser.add_argument("--text-path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--max-chars", type=int, default=150000)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument(
        "--quantile-sample-size",
        type=int,
        default=200_000,
        help="Per-tensor sample size for quantile estimation (0 = full tensor, slow).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--act-calib-chars", type=int, default=20000)
    parser.add_argument("--act-calib-seq-len", type=int, default=256)
    parser.add_argument(
        "--models",
        default="qwen3,minicpm,internlm,gemma3",
        help="Comma-separated model keys to run: qwen3,qwen3_30b_fp8,minicpm,minicpm4_0p5b,minicpm_o_4_5,step_audio_2_mini,step_audio_tts_3b,step_audio_editx,internlm,gemma3",
    )
    parser.add_argument(
        "--include-wa",
        action="store_true",
        help="Include weight+activation (W+A) quantization configs for W>=6bit.",
    )
    parser.add_argument(
        "--include-outlier0",
        action="store_true",
        help="Add outlier=0% ablations (i.e., quantize all values; no outlier preservation).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs whose output JSON already exists in --results-dir.",
    )
    parser.add_argument(
        "--refresh-samples",
        action="store_true",
        help="If an output JSON already exists, regenerate only the continuation samples and update the file (reuse existing PPL).",
    )
    parser.add_argument(
        "--config-set",
        choices=("full", "wa"),
        default="full",
        help="Which configs to run: full=SPEC configs (optionally +W+A), wa=BF16 + 6/8-bit W and W+A only.",
    )
    parser.add_argument(
        "--quantize-all-blocks",
        action="store_true",
        help="Ablation: quantize ALL transformer blocks (default: keep first/last blocks in BF16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config_set == "wa":
        args.include_wa = True
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts = [
        "萧子山看着眼前的虫洞，心中充满了",
        "文总说道：",
        "穿越到明朝之后，他们首先要解决的是",
        "临高启明计划的核心是",
    ]

    experiments: Dict[str, Dict[str, Any]] = {
        "qwen3": {
            "model_path": os.environ.get("QWEN3_MODEL_PATH", "/publicdata/huggingface.co/Qwen/Qwen3-1.7B/"),
            "use_fast_tokenizer": False,
            "configs": [
                ExperimentConfig("BF16 baseline", None, 1.0),
                ExperimentConfig("4-bit W (1.0%)", 4, 1.0),
                ExperimentConfig("5-bit W (0.1%)", 5, 0.1),
                ExperimentConfig("6-bit W (1.0%)", 6, 1.0),
                ExperimentConfig("6-bit W (0.1%)", 6, 0.1),
                ExperimentConfig("8-bit W (1.0%)", 8, 1.0),
            ],
        },
        "qwen3_30b_fp8": {
            "model_path": os.environ.get(
                "QWEN3_30B_FP8_MODEL_PATH",
                "/publicdata/huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            ),
            "use_fast_tokenizer": False,
            "torch_dtype": "auto",
            "device_map": "cuda",
            "baseline_tag": "fp8",
            "configs": [
                ExperimentConfig("FP8 baseline", None, 1.0),
                ExperimentConfig("6-bit W (1.0%)", 6, 1.0),
            ],
        },
        "minicpm": {
            "model_path": os.environ.get("MINICPM_MODEL_PATH", "/workspace/zhousc6@xiaopeng.com/MiniCPM-2B-sft-bf16/"),
            "use_fast_tokenizer": None,
            "configs": [
                ExperimentConfig("BF16 baseline", None, 1.0),
                ExperimentConfig("4-bit W (1.0%)", 4, 1.0),
                ExperimentConfig("6-bit W (1.0%)", 6, 1.0),
                ExperimentConfig("8-bit W (1.0%)", 8, 1.0),
            ],
        },
        "minicpm4_0p5b": {
            "model_path": os.environ.get("MINICPM4_0P5B_MODEL_PATH", "/workspace/zhousc6@xiaopeng.com/MiniCPM4-0.5B"),
            "use_fast_tokenizer": None,
            "configs": [
                ExperimentConfig("BF16 baseline", None, 1.0),
                ExperimentConfig("4-bit W (1.0%)", 4, 1.0),
                ExperimentConfig("6-bit W (1.0%)", 6, 1.0),
                ExperimentConfig("8-bit W (1.0%)", 8, 1.0),
            ],
        },
        "minicpm_o_4_5": {
            "model_path": os.environ.get(
                "MINICPMO_4_5_MODEL_PATH",
                "/workspace/zhousc6@xiaopeng.com/MiniCPM-o-4_5",
            ),
            "use_fast_tokenizer": None,
            "configs": [
                ExperimentConfig("BF16 baseline", None, 1.0),
                ExperimentConfig("4-bit W (1.0%)", 4, 1.0),
                ExperimentConfig("6-bit W (1.0%)", 6, 1.0),
                ExperimentConfig("8-bit W (1.0%)", 8, 1.0),
            ],
        },
        "step_audio_2_mini": {
            "model_path": os.environ.get(
                "STEP_AUDIO_2_MINI_MODEL_PATH",
                "/workspace/zhousc6@xiaopeng.com/Step-Audio-2-mini-text",
            ),
            "use_fast_tokenizer": None,
            # Text-only eval: use the Qwen2 text backbone only; keep configs small (model is large).
            "configs": [
                ExperimentConfig("BF16 baseline", None, 1.0),
                ExperimentConfig("6-bit W (1.0%)", 6, 1.0),
                ExperimentConfig("8-bit W (1.0%)", 8, 1.0),
            ],
        },
        "step_audio_tts_3b": {
            "model_path": os.environ.get(
                "STEP_AUDIO_TTS_3B_MODEL_PATH",
                "/publicdata/huggingface.co/models/stepfun-ai/Step-Audio-TTS-3B/main",
            ),
            "use_fast_tokenizer": None,
            "configs": [
                ExperimentConfig("BF16 baseline", None, 1.0),
                ExperimentConfig("6-bit W (1.0%)", 6, 1.0),
                ExperimentConfig("8-bit W (1.0%)", 8, 1.0),
            ],
        },
        "step_audio_editx": {
            "model_path": os.environ.get(
                "STEP_AUDIO_EDITX_MODEL_PATH",
                "/publicdata/huggingface.co/models/stepfun-ai/Step-Audio-EditX/main",
            ),
            "use_fast_tokenizer": None,
            "configs": [
                ExperimentConfig("BF16 baseline", None, 1.0),
                ExperimentConfig("4-bit W (1.0%)", 4, 1.0),
                ExperimentConfig(
                    "4-bit W + 6-bit A (1%)",
                    bits=4,
                    outlier_percentile=1.0,
                    act_bits=6,
                    act_outlier_percentile=1.0,
                ),
                ExperimentConfig("6-bit W (1.0%)", 6, 1.0),
                ExperimentConfig(
                    "6-bit W+A (0%)",
                    bits=6,
                    outlier_percentile=0.0,
                    act_bits=6,
                    act_outlier_percentile=0.0,
                ),
                ExperimentConfig(
                    "6-bit W(outlier=0%)+A(outlier=0.1%)",
                    bits=6,
                    outlier_percentile=0.0,
                    act_bits=6,
                    act_outlier_percentile=0.1,
                ),
                ExperimentConfig("8-bit W (1.0%)", 8, 1.0),
            ],
        },
        "internlm": {
            "model_path": _resolve_internlm_snapshot_path(
                os.environ.get(
                    "INTERNLM_MODEL_ROOT",
                    "/workspace/zhousc6@xiaopeng.com/models--internlm--internlm2_5-1_8b",
                )
            ),
            "use_fast_tokenizer": None,
            "configs": [
                ExperimentConfig("BF16 baseline", None, 1.0),
                ExperimentConfig("4-bit W (1.0%)", 4, 1.0),
                ExperimentConfig("6-bit W (1.0%)", 6, 1.0),
                ExperimentConfig("8-bit W (1.0%)", 8, 1.0),
            ],
        },
        "gemma3": {
            "model_path": os.environ.get("GEMMA3_MODEL_PATH", "/workspace/zhousc6@xiaopeng.com/gemma-3-1b-it/"),
            "use_fast_tokenizer": None,
            "configs": [
                ExperimentConfig("BF16 baseline", None, 1.0),
                ExperimentConfig("4-bit W (1.0%)", 4, 1.0),
                ExperimentConfig("6-bit W (1.0%)", 6, 1.0),
                ExperimentConfig("8-bit W (1.0%)", 8, 1.0),
            ],
        },
    }

    if args.include_outlier0:
        for spec in experiments.values():
            cfgs: List[ExperimentConfig] = list(spec["configs"])
            bits_present = sorted({int(c.bits) for c in cfgs if c.bits is not None})
            extra: List[ExperimentConfig] = []
            for bits in bits_present:
                extra.append(ExperimentConfig(f"{bits}-bit W (0%)", bits, 0.0))
            spec["configs"] = cfgs + extra

    if args.include_wa:
        for spec in experiments.values():
            cfgs: List[ExperimentConfig] = list(spec["configs"])
            extra: List[ExperimentConfig] = []
            for c in cfgs:
                if c.bits is None:
                    continue
                if int(c.bits) < 6:
                    continue
                if abs(float(c.outlier_percentile) - 1.0) > 1e-9:
                    # Keep W+A matrix small; only run the default outlier setting.
                    continue
                extra.append(
                    ExperimentConfig(
                        label=f"{int(c.bits)}-bit W+A ({c.outlier_percentile:g}%)",
                        bits=int(c.bits),
                        outlier_percentile=float(c.outlier_percentile),
                        act_bits=int(c.bits),
                        act_outlier_percentile=float(c.outlier_percentile),
                    )
                )
            if args.include_outlier0:
                for bits in sorted({int(c.bits) for c in cfgs if c.bits is not None and int(c.bits) >= 6}):
                    extra.append(
                        ExperimentConfig(
                            label=f"{bits}-bit W+A (0%)",
                            bits=bits,
                            outlier_percentile=0.0,
                            act_bits=bits,
                            act_outlier_percentile=0.0,
                        )
                    )
                    # Mixed outlier ablations to isolate which side matters (W vs activation).
                    extra.append(
                        ExperimentConfig(
                            label=f"{bits}-bit W(outlier=1%)+A(outlier=0%)",
                            bits=bits,
                            outlier_percentile=1.0,
                            act_bits=bits,
                            act_outlier_percentile=0.0,
                        )
                    )
                    extra.append(
                        ExperimentConfig(
                            label=f"{bits}-bit W(outlier=0%)+A(outlier=1%)",
                            bits=bits,
                            outlier_percentile=0.0,
                            act_bits=bits,
                            act_outlier_percentile=1.0,
                        )
                    )
            spec["configs"] = cfgs + extra

    if args.config_set == "wa":
        for spec in experiments.values():
            cfgs: List[ExperimentConfig] = list(spec["configs"])
            filtered: List[ExperimentConfig] = []
            for c in cfgs:
                if c.bits is None:
                    filtered.append(c)
                    continue
                if int(c.bits) not in (6, 8):
                    continue
                if abs(float(c.outlier_percentile) - 1.0) > 1e-9 and abs(float(c.outlier_percentile) - 0.0) > 1e-9:
                    continue
                # Keep both W-only and W+A for these bits.
                filtered.append(c)
            spec["configs"] = filtered

    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for model_key in model_keys:
        if model_key not in experiments:
            raise SystemExit(f"Unknown model key: {model_key}")

    all_results: List[Dict[str, Any]] = []
    for model_key in model_keys:
        spec = experiments[model_key]
        model_path = spec["model_path"]
        use_fast_tokenizer = spec["use_fast_tokenizer"]
        configs: List[ExperimentConfig] = spec["configs"]

        print("=" * 90)
        print(f"[Model] {model_key} | path={model_path}")
        print("=" * 90)

        for cfg in configs:
            cfg_run = cfg.with_prefix("[ALLBLOCKS] " if args.quantize_all_blocks else "")
            print(f"\n[Run] {cfg_run.label} | bits={cfg_run.bits} | outlier={cfg_run.outlier_percentile}%")
            scope_suffix = "_allblocks" if args.quantize_all_blocks else ""
            baseline_tag = spec.get("baseline_tag", "bf16")
            out_name = (
                f"{model_key}_{baseline_tag}{scope_suffix}.json"
                if cfg_run.bits is None
                else (
                    f"{model_key}_{cfg_run.bits}bit_outlier{cfg_run.outlier_percentile:g}{scope_suffix}.json"
                    if cfg_run.act_bits is None
                    else f"{model_key}_{cfg_run.bits}bit_outlier{cfg_run.outlier_percentile:g}_act{cfg_run.act_bits}bit_actoutlier{cfg_run.act_outlier_percentile:g}{scope_suffix}.json"
                )
            )
            out_path = results_dir / out_name
            existing_payload: Optional[Dict[str, Any]] = None
            if out_path.exists():
                if args.refresh_samples:
                    try:
                        existing_payload = json.loads(out_path.read_text(encoding="utf-8"))
                    except Exception:
                        existing_payload = None
                elif args.skip_existing:
                    print(f"[Skip] Exists: {out_path}")
                    continue

            payload = _run_single(
                model_key=model_key,
                model_path=model_path,
                use_fast_tokenizer=use_fast_tokenizer,
                text_path=args.text_path,
                max_chars=args.max_chars,
                results_dir=results_dir,
                prompts=prompts,
                n_ctx=args.n_ctx,
                stride=args.stride,
                config=cfg_run,
                quantize_all_blocks=bool(args.quantize_all_blocks),
                seed=args.seed,
                quantile_sample_size=args.quantile_sample_size,
                act_calib_chars=args.act_calib_chars,
                act_calib_seq_len=args.act_calib_seq_len,
                torch_dtype=spec.get("torch_dtype"),
                device_map=spec.get("device_map", "auto"),
                compute_ppl=(existing_payload is None),
                existing_ppl=(None if existing_payload is None else existing_payload.get("ppl")),
            )

            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            all_results.append(payload)
            print(f"[Done] PPL={payload['ppl']} -> {out_path}")

    summary_path = results_dir / "summary.json"
    merged_results: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            for item in existing:
                key = (
                    item.get("model_key"),
                    str(item.get("target_layer_mode") or "middle"),
                    item.get("num_bits"),
                    float(item.get("outlier_percentile", 1.0)),
                    item.get("act_bits", None),
                    float(item.get("act_outlier_percentile", 1.0)),
                )
                merged_results[key] = item
        except Exception:
            # If an existing summary is corrupted or incompatible, overwrite it with the new run.
            merged_results = {}

    for item in all_results:
        key = (
            item.get("model_key"),
            str(item.get("target_layer_mode") or "middle"),
            item.get("num_bits"),
            float(item.get("outlier_percentile", 1.0)),
            item.get("act_bits", None),
            float(item.get("act_outlier_percentile", 1.0)),
        )
        merged_results[key] = item

    final_results = list(merged_results.values())
    final_results.sort(
        key=lambda it: (
            str(it.get("model_key", "")),
            str(it.get("target_layer_mode") or "middle"),
            0 if it.get("num_bits") is None else 1,
            0 if it.get("num_bits") is None else int(it.get("num_bits") or 0),
            float(it.get("outlier_percentile", 1.0)),
            0 if it.get("act_bits") is None else 1,
            0 if it.get("act_bits") is None else int(it.get("act_bits") or 0),
            float(it.get("act_outlier_percentile", 1.0)),
        )
    )

    summary_path.write_text(json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
