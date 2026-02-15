# SPEC DoReFa Experiments + Report

This repo includes a lightweight, self-contained experiment pipeline for the `SPEC.md` DoReFa-style quantization
(keep first/last transformer blocks in BF16; quantize the middle blocks).

The pipeline does **not** depend on the `gptqmodel` Python package; it uses `torch` + `transformers` only.

## Quickstart

```bash
# 1) Run experiments (writes JSON to the run dir)
RUN_DIR="results/spec_run_$(date +%Y%m%d_%H%M%S)_wa"
python scripts/dorefa_spec_experiments.py \
  --results-dir "$RUN_DIR" \
  --models qwen3,minicpm,internlm,minicpm4_0p5b \
  --include-wa \
  --include-outlier0

# 2) Plot layer-wise error accumulation (default: exclude first/last block)
python scripts/plot_layer_quant_error.py --run-dir "$RUN_DIR"

# 3) Generate HTML report (exclude gemma if transformers lacks Gemma3 support)
python scripts/generate_spec_html_report.py --run-dir "$RUN_DIR" --exclude-models gemma3

# 4) Convert HTML -> PDF (pandoc expands <details> blocks, matching "folded sections expanded")
cd "$RUN_DIR"
pandoc report.html -o report.pdf \
  --pdf-engine=xelatex \
  -V mainfont="Noto Sans CJK SC" \
  -V monofont="Noto Sans Mono CJK SC"
```

## Notes

- `scripts/dorefa_spec_experiments.py` supports:
  - Weight-only quantization (W).
  - Weight+activation quantization (W+A) via forward hooks (enabled by `--include-wa`, only for W>=6bit).
  - Outlier ablations, including `0%` outlier (enabled by `--include-outlier0`).
- `scripts/plot_layer_quant_error.py` defaults to **not counting the first/last transformer blocks** in plots.
  Use `--include-ends` to include them.

