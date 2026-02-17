#!/usr/bin/env python3
"""
Simplified test script for Qwen3-30B-A3B-Instruct-2507-FP8 model.
Tests FP8 baseline and 6-bit quantization with PPL and generation.
"""

import os
import sys
import json
import gc
import math
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

DEFAULT_TEXT_PATH = "/workspace/zhousc6@xiaopeng.com/swift_train/lgqm.txt"
MODEL_PATH = "/publicdata/huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
RESULTS_DIR = Path("results/qwen3_30b_a3b_fp8")

def load_text(text_path: str, max_chars: int = 50000) -> str:
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text[:max_chars] if len(text) > max_chars else text

def load_model_and_tokenizer(model_path: str):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading model (this may take a few minutes)...")
    # Try loading with different strategies
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("Model loaded with device_map='auto'")
    except Exception as e:
        print(f"device_map='auto' failed: {e}")
        print("Trying fallback...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.cuda()
        print("Model loaded to CUDA")
    
    model.eval()
    return model, tokenizer

def infer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise AttributeError("Cannot find transformer layers")

def iter_target_weights(model, target_layer_indices):
    layers = infer_layers(model)
    for layer_idx in target_layer_indices:
        layer = layers[layer_idx]
        
        # Attention projections
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    yield f"layer{layer_idx}.self_attn.{proj_name}.weight", proj.weight
        
        # MoE experts
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            experts = getattr(mlp, "experts", None)
            if experts is not None:
                for expert_idx, expert in enumerate(experts):
                    for proj_name in ("gate_proj", "up_proj", "down_proj"):
                        proj = getattr(expert, proj_name, None)
                        if proj is not None and hasattr(proj, "weight"):
                            yield f"layer{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight", proj.weight
            
            shared_expert = getattr(mlp, "shared_expert", None)
            if shared_expert is not None:
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    proj = getattr(shared_expert, proj_name, None)
                    if proj is not None and hasattr(proj, "weight"):
                        yield f"layer{layer_idx}.mlp.shared_expert.{proj_name}.weight", proj.weight

def compute_quantile_bounds(weight, outlier_percentile, sample_size=200000):
    q = outlier_percentile / 100.0
    wf = weight.detach().float().reshape(-1)
    if sample_size > 0 and wf.numel() > sample_size:
        step = max(1, wf.numel() // sample_size)
        wf = wf[::step][:sample_size]
    sample = wf.detach().cpu().numpy()
    w_min = float(np.quantile(sample, q))
    w_max = float(np.quantile(sample, 1.0 - q))
    return w_min, w_max

def dorefa_quantize_inplace(weight, num_bits, bounds):
    qmin = -(2 ** (num_bits - 1) - 1)
    qmax = 2 ** (num_bits - 1) - 1
    w_min, w_max = bounds
    wf = weight.detach().float()
    outlier_mask = (wf < w_min) | (wf > w_max)
    scale = (w_max / qmax) if w_max > 0 else 1.0
    wq = torch.clamp(torch.round(wf / scale), qmin, qmax) * scale
    wq = wq.to(dtype=weight.dtype)
    weight.data.copy_(torch.where(outlier_mask, weight.data, wq))

def quantize_weights(model, target_layer_indices, bits, outlier_percentile=1.0):
    print(f"Quantizing weights to {bits}-bit (outlier_percentile={outlier_percentile}%)...")
    count = 0
    for name, weight in iter_target_weights(model, target_layer_indices):
        bounds = compute_quantile_bounds(weight, outlier_percentile)
        dorefa_quantize_inplace(weight, bits, bounds)
        count += 1
    print(f"Quantized {count} weight tensors")

@torch.inference_mode()
def calculate_ppl(model, tokenizer, text, n_ctx=2048, stride=1024):
    encodings = tokenizer(text, truncation=False, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    seq_len = input_ids.size(1)
    
    total_nll = 0.0
    total_tokens = 0
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + n_ctx, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100
        
        outputs = model(input_chunk, labels=target_ids, use_cache=False)
        loss = outputs.loss
        
        if loss is not None:
            total_nll += (loss.float().item() * trg_len)
            total_tokens += trg_len
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)

@torch.inference_mode()
def generate_samples(model, tokenizer, prompts, max_new_tokens=80):
    samples = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        continuation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        samples.append({"prompt": prompt, "continuation": continuation})
    return samples

def run_experiment(config_name, bits=None, max_chars=50000, n_ctx=2048, stride=1024):
    print(f"\n{'='*80}")
    print(f"Running: {config_name}")
    print(f"{'='*80}")
    
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    layers = infer_layers(model)
    target_layers = list(range(1, len(layers) - 1))  # Skip first and last layer
    
    print(f"Model has {len(layers)} layers, targeting {len(target_layers)} layers")
    
    if bits is not None:
        quantize_weights(model, target_layers, bits)
    
    print("Loading text for PPL calculation...")
    text = load_text(DEFAULT_TEXT_PATH, max_chars)
    
    print(f"Calculating PPL (n_ctx={n_ctx}, stride={stride})...")
    ppl = calculate_ppl(model, tokenizer, text, n_ctx, stride)
    print(f"PPL: {ppl:.4f}")
    
    prompts = [
        "萧子山看着眼前的虫洞，心中充满了",
        "文总说道：",
        "穿越到明朝之后，他们首先要解决的是",
        "临高启明计划的核心是",
    ]
    print("Generating samples...")
    samples = generate_samples(model, tokenizer, prompts)
    
    result = {
        "config": config_name,
        "bits": bits,
        "ppl": round(ppl, 6),
        "samples": samples,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=50000)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--configs", default="fp8,6", help="Comma-separated: fp8,6,6w+a")
    args = parser.parse_args()
    
    configs = []
    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        if cfg == "fp8":
            configs.append(("FP8 baseline", None))
        elif cfg == "6":
            configs.append(("6-bit W", 6))
        elif cfg == "6w+a":
            configs.append(("6-bit W+A", 6))  # Will be handled separately
    
    all_results = []
    for name, bits in configs:
        try:
            result = run_experiment(name, bits, args.max_chars, args.n_ctx, args.stride)
            all_results.append(result)
            
            # Save individual result
            out_path = RESULTS_DIR / f"result_{name.replace(' ', '_').replace('+', 'plus')}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Saved to {out_path}")
        except Exception as e:
            print(f"Error running {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
