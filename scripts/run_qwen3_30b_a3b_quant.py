#!/usr/bin/env python3
"""
Run the DoReFa weight-only experiments for Qwen3-30B-A3B-Instruct-2507 model.

Using GPU 4,5 for the experiment.
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


DEFAULT_TEXT_PATH = "/workspace/zhousc6@xiaopeng.com/swift_train/lgqm.txt"


@dataclass(frozen=True)
class ExperimentConfig:
    label: str
    bits: Optional[int]
    outlier_percentile: float = 1.0
    act_bits: Optional[int] = None
    act_outlier_percentile: float = 1.0


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

        # MoE support: also quantize experts
        moe = getattr(layer, "mlp", None)
        if moe is not None and hasattr(moe, "experts"):
            # Qwen3 MoE uses experts list
            experts = getattr(moe, "experts", None)
            if experts is not None:
                for expert_idx, expert in enumerate(experts):
                    for proj_name in ("gate_proj", "up_proj", "down_proj"):
                        proj = getattr(expert, proj_name, None)
                        if proj is None or not hasattr(proj, "weight"):
                            continue
                        yield f"layer{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight", proj.weight
            # Some MoE implementations use w1, w2, w3
            elif hasattr(moe, "w1"):
                for proj_name in ("w1", "w2", "w3"):
                    proj = getattr(moe, proj_name, None)
                    if proj is not None and hasattr(proj, "weight"):
                        yield f"layer{layer_idx}.mlp.{proj_name}.weight", proj.weight

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
    q = outlier_percentile / 100.0
    wf = weight.detach().float().reshape(-1)
    if sample_size > 0 and wf.numel() > sample_size:
        step = max(1, wf.numel() // sample_size)
        wf = wf[::step][:sample_size]

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
    hidden_states = list(out.hidden_states or [])
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
    samples: List[Dict[str, str]] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        )
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
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


def _load_model_and_tokenizer(model_path: str, *, use_fast: Optional[bool]) -> Tuple[Any, Any]:
    tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if use_fast is not None:
        tokenizer_kwargs["use_fast"] = use_fast
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    # For large MoE models, try different loading strategies
    model = None
    
    # Strategy 1: Try device_map="auto" without tp_plan
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            tp_plan=None,
            **model_kwargs
        )
        print(f"  Model loaded with device_map='auto'")
    except Exception as e:
        print(f"  Failed to load with device_map='auto': {e}")
    
    # Strategy 2: Try with max_memory to limit to available GPUs
    if model is None:
        try:
            max_memory = {0: "70GiB", 1: "70GiB", 2: "70GiB", 3: "70GiB"}
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                max_memory=max_memory,
                **model_kwargs
            )
            print(f"  Model loaded with max_memory constraint")
        except Exception as e2:
            print(f"  Failed to load with max_memory: {e2}")
    
    # Strategy 3: Try low_cpu_mem_usage mode
    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                **model_kwargs
            )
            print(f"  Model loaded with low_cpu_mem_usage")
        except Exception as e3:
            print(f"  Failed to load with low_cpu_mem_usage: {e3}")
    
    # Strategy 4: Fallback to CPU load then move
    if model is None:
        print("  Trying CPU load fallback...")
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if torch.cuda.is_available():
            # Move to CUDA if possible
            try:
                model = model.cuda()
                print(f"  Model loaded to CUDA manually")
            except Exception as e4:
                print(f"  Failed to move to CUDA, keeping on CPU: {e4}")
        else:
            print(f"  Model loaded on CPU")

    model.eval()
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
    compute_ppl: bool = True,
    existing_ppl: Optional[float] = None,
) -> Dict[str, Any]:
    _seed_all(seed)
    model, tokenizer = _load_model_and_tokenizer(model_path, use_fast=use_fast_tokenizer)

    layers = _infer_layers(model)
    if quantize_all_blocks:
        target_layers = list(range(0, len(layers)))
        target_layer_mode = "all"
    else:
        target_layers = list(range(1, len(layers) - 1))
        target_layer_mode = "middle"

    print(f"  Model has {len(layers)} layers, targeting {len(target_layers)} layers for quantization")

    if config.bits is not None:
        scope_suffix = "_allblocks" if quantize_all_blocks else ""
        bounds_cache_path = (
            results_dir
            / "quantile_cache"
            / f"{model_key}_outlier{config.outlier_percentile:g}{scope_suffix}.json"
        )

        print(f"  Computing quantile bounds (outlier_percentile={config.outlier_percentile}%)...")
        bounds = _load_or_build_bounds_cache(
            cache_path=bounds_cache_path,
            model=model,
            target_layer_indices=target_layers,
            outlier_percentile=config.outlier_percentile,
            sample_size=quantile_sample_size,
        )
        print(f"  Quantizing {len(bounds)} weight tensors to {config.bits}-bit...")
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
        print(f"  Computing PPL (n_ctx={n_ctx}, stride={stride})...")
        text = _load_text(text_path, max_chars=max_chars)
        ppl = _calculate_ppl(model, tokenizer, text, n_ctx=n_ctx, stride=stride)
    else:
        ppl = float(existing_ppl) if existing_ppl is not None else float("nan")
    
    print(f"  Generating samples...")
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
        default=Path("results/qwen3_30b_a3b"),
        help="Directory to write JSON results and caches.",
    )
    parser.add_argument("--text-path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--max-chars", type=int, default=100000, help="Text length for PPL eval (reduce for large models)")
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
        "--skip-existing",
        action="store_true",
        help="Skip configs whose output JSON already exists in --results-dir.",
    )
    parser.add_argument(
        "--quantize-all-blocks",
        action="store_true",
        help="Ablation: quantize ALL transformer blocks (default: keep first/last blocks in BF16).",
    )
    parser.add_argument(
        "--include-wa",
        action="store_true",
        help="Include weight+activation (W+A) quantization configs for W>=6bit.",
    )
    parser.add_argument(
        "--model-path",
        default="/publicdata/huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Path to Qwen3-30B-A3B-Instruct-2507 model",
    )
    parser.add_argument(
        "--configs",
        default="4,6,8",
        help="Comma-separated list of bit widths to test (e.g., '4,6,8' or 'bf16,4,6,8')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts = [
        "萧子山看着眼前的虫洞，心中充满了",
        "文总说道：",
        "穿越到明朝之后，他们首先要解决的是",
        "临高启明计划的核心是",
    ]

    # Parse config bits
    config_bits = []
    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        if cfg.lower() == "bf16" or cfg.lower() == "baseline":
            config_bits.append(None)
        else:
            config_bits.append(int(cfg))

    # Build experiment configs
    configs = []
    for bits in config_bits:
        if bits is None:
            configs.append(ExperimentConfig("BF16 baseline", None, 1.0))
        else:
            configs.append(ExperimentConfig(f"{bits}-bit W (1.0%)", bits, 1.0))
            # Add 0.1% outlier variant for 5+ bits
            if bits >= 5:
                configs.append(ExperimentConfig(f"{bits}-bit W (0.1%)", bits, 0.1))

    # Add W+A configs if requested
    if args.include_wa:
        for bits in [6, 8]:
            configs.append(ExperimentConfig(
                f"{bits}-bit W+A (1.0%)", 
                bits, 1.0, 
                act_bits=bits, 
                act_outlier_percentile=1.0
            ))

    model_key = "qwen3_30b_a3b"
    model_path = args.model_path
    use_fast_tokenizer = False

    print("=" * 90)
    print(f"[Model] {model_key} | path={model_path}")
    print(f"[GPUs] {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"[Configs] {len(configs)} experiment(s)")
    print("=" * 90)

    all_results: List[Dict[str, Any]] = []
    for cfg in configs:
        cfg_run = cfg
        print(f"\n[Run] {cfg_run.label} | bits={cfg_run.bits} | outlier={cfg_run.outlier_percentile}%")
        
        scope_suffix = "_allblocks" if args.quantize_all_blocks else ""
        out_name = (
            f"{model_key}_bf16{scope_suffix}.json"
            if cfg_run.bits is None
            else (
                f"{model_key}_{cfg_run.bits}bit_outlier{cfg_run.outlier_percentile:g}{scope_suffix}.json"
                if cfg_run.act_bits is None
                else f"{model_key}_{cfg_run.bits}bit_outlier{cfg_run.outlier_percentile:g}_act{cfg_run.act_bits}bit_actoutlier{cfg_run.act_outlier_percentile:g}{scope_suffix}.json"
            )
        )
        out_path = results_dir / out_name

        if out_path.exists() and args.skip_existing:
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
            compute_ppl=True,
            existing_ppl=None,
        )

        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        all_results.append(payload)
        print(f"[Done] PPL={payload['ppl']} -> {out_path}")

    # Write summary
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
