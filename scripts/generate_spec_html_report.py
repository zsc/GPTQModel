#!/usr/bin/env python3
"""Generate an HTML report from a SPEC DoReFa experiment run.

Input:
- <run_dir>/summary.json produced by `scripts/dorefa_spec_experiments.py`

Output:
- <run_dir>/report.html by default
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class Row:
    model_key: str
    model_path: str
    config: str
    num_bits: Optional[int]
    outlier_percentile: float
    act_bits: Optional[int]
    act_outlier_percentile: float
    ppl: float
    quantile_sample_size: int
    samples: List[dict]


def _safe(text: str) -> str:
    return html.escape(text, quote=True)


def _format_ppl(value: float) -> str:
    if not math.isfinite(value):
        return "inf"
    if abs(value) >= 1e6:
        return f"{value:.3e}"
    return f"{value:.4f}"


def _format_pct(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.2f}%"


def _diff_color(diff_pct: float) -> str:
    if not math.isfinite(diff_pct):
        return "#6c757d"
    abs_diff = abs(diff_pct)
    if abs_diff < 0.5:
        return "#198754"  # green
    if abs_diff < 2.0:
        return "#b78103"  # amber
    return "#dc3545"  # red


def _find_row(
    model_rows: List[Row],
    *,
    bits: int,
    outlier: float,
    act_bits: Optional[int],
    act_outlier: float = 1.0,
) -> Optional[Row]:
    eps = 1e-9
    for r in model_rows:
        if r.num_bits != bits:
            continue
        if abs(r.outlier_percentile - outlier) > eps:
            continue
        if act_bits is None:
            if r.act_bits is None:
                return r
            continue
        if r.act_bits != act_bits:
            continue
        if abs(r.act_outlier_percentile - act_outlier) > eps:
            continue
        return r
    return None


def _render_overview(by_model: Dict[str, List[Row]]) -> str:
    """Top-level summary table across models for quick scanning."""
    cols = [
        ("BF16", None),
        ("W8 (1%)", {"bits": 8, "outlier": 1.0, "act_bits": None, "act_outlier": 1.0}),
        ("W6 (1%)", {"bits": 6, "outlier": 1.0, "act_bits": None, "act_outlier": 1.0}),
        ("W8+A8 (1%)", {"bits": 8, "outlier": 1.0, "act_bits": 8, "act_outlier": 1.0}),
        ("W6+A6 (1%)", {"bits": 6, "outlier": 1.0, "act_bits": 6, "act_outlier": 1.0}),
    ]

    parts: List[str] = []
    parts.append("<section class='overview'>")
    parts.append("<h2>Overview</h2>")
    parts.append(
        "<p class='muted'>Quick PPL summary (lower is better). Full configs and samples are in per-model sections below.</p>"
    )
    parts.append("<table class='overview-table'>")
    parts.append("<thead><tr><th>Model</th>" + "".join([f"<th>{_safe(c[0])}</th>" for c in cols]) + "</tr></thead>")
    parts.append("<tbody>")

    for model_key in sorted(by_model.keys()):
        model_rows = by_model[model_key]
        baseline = _pick_baseline(model_rows)
        base_ppl = baseline.ppl if baseline is not None else float("nan")

        cells: List[str] = []
        for label, spec in cols:
            if spec is None:
                if baseline is None:
                    cells.append("<td class='mono muted'>n/a</td>")
                else:
                    cells.append(f"<td class='mono'>{_format_ppl(baseline.ppl)}</td>")
                continue

            row = _find_row(model_rows, **spec)
            if row is None:
                cells.append("<td class='mono muted'>-</td>")
                continue

            diff_pct = (
                ((row.ppl - base_ppl) / base_ppl * 100.0)
                if math.isfinite(base_ppl) and base_ppl != 0
                else float("nan")
            )
            diff_html = f"<span style='color:{_diff_color(diff_pct)}'>{_format_pct(diff_pct)}</span>"
            cells.append(f"<td class='mono'>{_format_ppl(row.ppl)}<div class='muted'>{diff_html}</div></td>")

        parts.append("<tr>" + f"<td class='modelkey'><a href='#{_safe(model_key)}'>{_safe(model_key)}</a></td>" + "".join(cells) + "</tr>")

    parts.append("</tbody></table>")
    parts.append(
        "<div class='muted' style='margin-top:10px'>"
        "Links: <a href='layer_error/index.html'><code>layer_error/index.html</code></a> | "
        "<a href='summary.json'><code>summary.json</code></a>"
        "</div>"
    )
    parts.append("</section>")
    return "\n".join(parts)


def _latest_run_dir(results_root: Path) -> Path:
    candidates = sorted(results_root.glob("spec_run_*"))
    if not candidates:
        raise SystemExit(f"No spec_run_* directories found under: {results_root}")
    return candidates[-1]


def _read_rows(summary_path: Path) -> List[Row]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: List[Row] = []
    for item in payload:
        rows.append(
            Row(
                model_key=str(item.get("model_key", "")),
                model_path=str(item.get("model_path", "")),
                config=str(item.get("config", "")),
                num_bits=item.get("num_bits", None),
                outlier_percentile=float(item.get("outlier_percentile", 1.0)),
                act_bits=item.get("act_bits", None),
                act_outlier_percentile=float(item.get("act_outlier_percentile", 1.0)),
                ppl=float(item.get("ppl", float("nan"))),
                quantile_sample_size=int(item.get("quantile_sample_size", 0)),
                samples=list(item.get("samples") or []),
            )
        )
    return rows


def _pick_baseline(model_rows: List[Row]) -> Optional[Row]:
    baseline = next((r for r in model_rows if r.num_bits is None), None)
    if baseline is not None:
        return baseline
    return next((r for r in model_rows if "bf16" in r.config.lower() or "baseline" in r.config.lower()), None)


def _short_config_name(row: Row) -> str:
    if row.num_bits is None:
        return "BF16"
    name = f"W{int(row.num_bits)} ({row.outlier_percentile:g}%)"
    if row.act_bits is not None:
        name += f" + A{int(row.act_bits)} ({row.act_outlier_percentile:g}%)"
    return name


def _select_rows_for_samples(model_key: str, model_rows: List[Row]) -> List[Row]:
    # SPEC requirement: every PPL record must include aligned continuation samples
    # (same prompts) for apples-to-apples comparison across configs.
    _ = model_key
    return sorted(
        model_rows,
        key=lambda r: (
            r.num_bits is not None,
            0 if r.num_bits is None else int(r.num_bits),
            float(r.outlier_percentile),
            0 if r.act_bits is None else 1,
            0 if r.act_bits is None else int(r.act_bits),
            float(r.act_outlier_percentile),
            r.config,
        ),
    )


def _render_samples_table(row: Row) -> str:
    if not row.samples:
        return "<p class='muted'>No samples in this record.</p>"

    parts: List[str] = []
    parts.append("<table class='samples'><thead><tr><th>Prompt</th><th>Continuation</th></tr></thead><tbody>")
    for sample in row.samples:
        prompt = _safe(str(sample.get("prompt", "")))
        continuation = _safe(str(sample.get("continuation", "")))
        parts.append(
            "<tr>"
            f"<td class='prompt'>{prompt}</td>"
            f"<td class='continuation'><pre>{continuation}</pre></td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _render_model_section(run_dir: Path, model_key: str, model_rows: List[Row]) -> str:
    baseline = _pick_baseline(model_rows)
    baseline_ppl = baseline.ppl if baseline is not None else float("nan")

    rows_sorted = _select_rows_for_samples(model_key, model_rows)

    table_lines: List[str] = []
    for r in rows_sorted:
        diff_pct = (
            ((r.ppl - baseline_ppl) / baseline_ppl * 100.0)
            if math.isfinite(baseline_ppl) and baseline_ppl != 0
            else float("nan")
        )
        act_cell = "-"
        if r.act_bits is not None:
            act_cell = f"A{int(r.act_bits)} ({r.act_outlier_percentile:g}%)"
        table_lines.append(
            "<tr>"
            f"<td class='config'>{_safe(r.config)}</td>"
            f"<td class='bits'>{'BF16' if r.num_bits is None else r.num_bits}</td>"
            f"<td class='outlier'>{'-' if r.num_bits is None else f'{r.outlier_percentile:g}%'} </td>"
            f"<td class='act'>{_safe(act_cell)}</td>"
            f"<td class='ppl'>{_format_ppl(r.ppl)}</td>"
            f"<td class='diff' style='color:{_diff_color(diff_pct)}'>{_format_pct(diff_pct)}</td>"
            "</tr>"
        )

    sample_rows = rows_sorted
    sample_blocks: List[str] = []
    for r in sample_rows:
        diff_pct = (
            ((r.ppl - baseline_ppl) / baseline_ppl * 100.0)
            if math.isfinite(baseline_ppl) and baseline_ppl != 0
            else float("nan")
        )
        subtitle = f"{_short_config_name(r)} | PPL {_format_ppl(r.ppl)} ({_format_pct(diff_pct)})"
        sample_blocks.append(
            "<details class='sample-block'>"
            f"<summary>{_safe(subtitle)}</summary>"
            f"{_render_samples_table(r)}"
            "</details>"
        )

    # Optional layer error artifact (produced by scripts/plot_layer_quant_error.py)
    layer_png = Path("layer_error") / f"layer_error_{model_key}.png"
    layer_json = Path("layer_error") / f"layer_error_{model_key}.json"
    layer_png_abs = run_dir / layer_png
    layer_json_abs = run_dir / layer_json
    if layer_png_abs.exists():
        layer_html = (
            "<details class='layer-block' open>"
            "<summary>Layer-wise error accumulation (exclude first/last layers by default)</summary>"
            "<div class='layer-links'>"
            "Index: <a href='layer_error/index.html'><code>layer_error/index.html</code></a> | "
            f"JSON: <a href='{_safe(str(layer_json))}'><code>{_safe(str(layer_json))}</code></a>"
            "</div>"
            f"<div class='layer-plot'><img src='{_safe(str(layer_png))}' alt='layer error {model_key}' /></div>"
            "</details>"
        )
    else:
        hint = (
            "Missing layer error plot. Run: "
            "<code>python scripts/plot_layer_quant_error.py --run-dir "
            + _safe(str(run_dir))
            + "</code>"
        )
        layer_html = f"<p class='muted'>{hint}</p>"

    model_path = _safe(rows_sorted[0].model_path) if rows_sorted else ""
    meta_bits = baseline.quantile_sample_size if baseline is not None else 0
    meta = []
    if model_path:
        meta.append(f"Model path: <code>{model_path}</code>")
    if meta_bits:
        meta.append(f"Quantile sample size: <code>{meta_bits}</code>")

    return (
        "<section class='model'>"
        f"<h2>{_safe(model_key)}</h2>"
        f"<div class='meta'>{' | '.join(meta)}</div>"
        "<table class='results'>"
        "<thead><tr><th>Config</th><th>Bits</th><th>Outlier</th><th>Act</th><th>PPL</th><th>vs BF16</th></tr></thead>"
        "<tbody>"
        + "\n".join(table_lines)
        + "</tbody></table>"
        "<h3>Layer Error</h3>"
        + layer_html
        + "<h3>Samples</h3>"
        + ("\n".join(sample_blocks) if sample_blocks else "<p class='muted'>No samples selected.</p>")
        + "</section>"
    )


def _render_html(run_dir: Path, rows: List[Row], *, exclude_models: Optional[Set[str]] = None) -> str:
    by_model: Dict[str, List[Row]] = {}
    for r in rows:
        if exclude_models is not None and r.model_key in exclude_models:
            continue
        by_model.setdefault(r.model_key, []).append(r)

    overview_html = _render_overview(by_model) if by_model else ""
    sections = [
        _render_model_section(run_dir, model_key, by_model[model_key])
        for model_key in sorted(by_model.keys())
    ]

    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width,initial-scale=1' />
  <title>SPEC DoReFa Report - {_safe(run_dir.name)}</title>
  <style>
    :root {{
      --bg: #0b0f17;
      --panel: #0f1623;
      --panel2: #121b2b;
      --text: #e7ecf5;
      --muted: #9aa7bf;
      --line: rgba(231,236,245,0.12);
      --accent: #5dd6c7;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Liberation Sans", sans-serif;
    }}
    body {{
      margin: 0;
      font-family: var(--sans);
      background: radial-gradient(1200px 700px at 10% 10%, rgba(122,162,255,0.16), transparent 60%),
                  radial-gradient(900px 600px at 90% 25%, rgba(93,214,199,0.14), transparent 55%),
                  var(--bg);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 18px 80px;
    }}
    header {{
      padding: 20px 18px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(18,27,43,0.9), rgba(15,22,35,0.9));
      border-radius: 16px;
      box-shadow: 0 18px 40px rgba(0,0,0,0.35);
    }}
    header h1 {{
      margin: 0 0 6px 0;
      font-size: 22px;
    }}
    header .sub {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    code {{
      font-family: var(--mono);
      font-size: 0.95em;
      background: rgba(231,236,245,0.08);
      border: 1px solid var(--line);
      padding: 1px 6px;
      border-radius: 8px;
    }}
    .model {{
      margin-top: 18px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(18,27,43,0.80), rgba(15,22,35,0.85));
    }}
    .overview {{
      margin-top: 18px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(18,27,43,0.78), rgba(15,22,35,0.82));
    }}
    .overview h2 {{
      margin: 0 0 8px 0;
      font-size: 16px;
      display: inline-block;
      padding-bottom: 6px;
      border-bottom: 2px solid rgba(122,162,255,0.35);
    }}
    .overview-table th:nth-child(1) {{ width: 18%; }}
    .overview-table td.modelkey a {{ color: var(--text); text-decoration: none; }}
    .overview-table td.modelkey a:hover {{ text-decoration: underline; }}
    td.mono {{ font-family: var(--mono); }}
    .model h2 {{
      margin: 0 0 6px 0;
      font-size: 18px;
      display: inline-block;
      padding-bottom: 6px;
      border-bottom: 2px solid rgba(93,214,199,0.45);
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      margin: 2px 0 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: rgba(15,22,35,0.6);
    }}
    th, td {{
      padding: 10px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      text-align: left;
      background: rgba(122,162,255,0.12);
    }}
    tr:hover td {{
      background: rgba(93,214,199,0.06);
    }}
    td.config {{ width: 44%; }}
    td.bits, td.outlier, td.ppl, td.diff {{ font-family: var(--mono); }}
    td.bits {{ width: 8%; }}
    td.outlier {{ width: 10%; }}
    td.act {{ width: 13%; font-family: var(--mono); }}
    td.ppl {{ width: 14%; }}
    td.diff {{ width: 12%; font-weight: 700; }}
    h3 {{
      margin: 18px 0 10px;
      font-size: 14px;
      color: rgba(231,236,245,0.92);
    }}
    .muted {{
      color: var(--muted);
      font-size: 12px;
    }}
    details.layer-block {{
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(15,22,35,0.55);
      overflow: hidden;
    }}
    details.layer-block > summary {{
      cursor: pointer;
      padding: 10px 12px;
      font-size: 13px;
      background: rgba(122,162,255,0.10);
      border-bottom: 1px solid var(--line);
      list-style: none;
    }}
    details.layer-block > summary::-webkit-details-marker {{ display: none; }}
    .layer-links {{
      color: var(--muted);
      font-size: 12px;
      padding: 10px 12px 0;
    }}
    .layer-plot {{
      padding: 10px 12px 12px;
    }}
    .layer-plot img {{
      max-width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: rgba(231,236,245,0.04);
      display: block;
    }}
    details.sample-block {{
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(15,22,35,0.55);
      overflow: hidden;
    }}
    details.sample-block > summary {{
      cursor: pointer;
      padding: 10px 12px;
      font-size: 13px;
      background: rgba(93,214,199,0.08);
      border-bottom: 1px solid var(--line);
      list-style: none;
    }}
    details.sample-block > summary::-webkit-details-marker {{ display: none; }}
    table.samples th:nth-child(1) {{ width: 28%; }}
    table.samples td.continuation pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.35;
      background: rgba(231,236,245,0.05);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 10px;
      max-height: 240px;
      overflow: auto;
    }}
    footer {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class='wrap'>
    <header>
      <h1>LLM 量化实验报告 (SPEC DoReFa)</h1>
      <p class='sub'>
        Run dir: <code>{_safe(str(run_dir))}</code><br />
        Generated at: <code>{_safe(generated_at)}</code><br />
        Source: <code>{_safe(str(run_dir / 'summary.json'))}</code>
      </p>
    </header>
    {overview_html}
    {''.join(sections)}
    <footer>
      Tip: “Samples” 只选取 BF16 + 常见 bit 配置做对比；完整数据在同目录 JSON。
    </footer>
  </div>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Experiment run directory containing summary.json (default: latest results/spec_run_*).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory to search for spec_run_* (default: results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <run-dir>/report.html).",
    )
    parser.add_argument(
        "--exclude-models",
        default="",
        help="Comma-separated model keys to exclude from report (e.g., gemma3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or _latest_run_dir(args.results_root)

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json: {summary_path}")

    rows = _read_rows(summary_path)
    exclude = {m.strip() for m in str(args.exclude_models).split(",") if m.strip()}
    html_text = _render_html(run_dir, rows, exclude_models=exclude or None)

    out_path = args.output or (run_dir / "report.html")
    out_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
