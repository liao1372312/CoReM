"""Dependency-free statistics, tables, and plotting helpers for experiments."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else float("nan")


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ordered = sorted(zip(probabilities, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, float]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    predictions = [int(value >= threshold) for value in probabilities]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    clipped = [min(1.0 - 1e-12, max(1e-12, value)) for value in probabilities]
    return {
        "n": float(len(labels)),
        "positive_rate": mean(float(value) for value in labels),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "roc_auc": roc_auc(labels, probabilities),
        "brier": mean((probability - label) ** 2 for label, probability in zip(labels, probabilities)),
        "log_loss": -mean(
            label * math.log(probability) + (1 - label) * math.log(1.0 - probability)
            for label, probability in zip(labels, clipped)
        ),
        "ece_10": expected_calibration_error(labels, probabilities, bins=10),
    }


def expected_calibration_error(
    labels: Sequence[int], probabilities: Sequence[float], *, bins: int
) -> float:
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, value in enumerate(probabilities)
            if lower <= value < upper or (index == bins - 1 and value == 1.0)
        ]
        if not members:
            continue
        observed = mean(float(labels[position]) for position in members)
        predicted = mean(probabilities[position] for position in members)
        error += len(members) / total * abs(observed - predicted)
    return error


def calibration_bins(
    labels: Sequence[int], probabilities: Sequence[float], *, bins: int = 10
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, value in enumerate(probabilities)
            if lower <= value < upper or (index == bins - 1 and value == 1.0)
        ]
        if not members:
            continue
        rows.append(
            {
                "bin": index + 1,
                "lower": lower,
                "upper": upper,
                "n": len(members),
                "mean_probability": mean(probabilities[position] for position in members),
                "observed_rate": mean(float(labels[position]) for position in members),
            }
        )
    return rows


def select_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    candidates = [index / 100.0 for index in range(5, 96)]
    return max(
        candidates,
        key=lambda threshold: (
            binary_metrics(labels, probabilities, threshold)["balanced_accuracy"],
            -abs(threshold - 0.5),
        ),
    )


def clustered_bootstrap_ci(
    rows: Sequence[dict[str, Any]],
    *,
    cluster_key: str,
    statistic: Callable[[Sequence[dict[str, Any]]], float],
    draws: int,
    seed: int,
) -> tuple[float, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(row)
    clusters = sorted(grouped)
    if not clusters:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled: list[dict[str, Any]] = []
        for key in rng.choices(clusters, k=len(clusters)):
            sampled.extend(grouped[key])
        estimates.append(statistic(sampled))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_randomization_pvalue(
    differences: Sequence[float], *, draws: int = 20_000, seed: int = 2027
) -> float:
    nonzero = [value for value in differences if abs(value) > 1e-15]
    if not nonzero:
        return 1.0
    observed = abs(mean(nonzero))
    if len(nonzero) <= 18:
        exceed = 0
        total = 1 << len(nonzero)
        for mask in range(total):
            value = mean(
                difference if mask & (1 << index) else -difference
                for index, difference in enumerate(nonzero)
            )
            exceed += abs(value) >= observed - 1e-15
        return exceed / total
    rng = random.Random(seed)
    exceed = 0
    for _ in range(draws):
        value = mean(rng.choice((-1.0, 1.0)) * difference for difference in nonzero)
        exceed += abs(value) >= observed - 1e-15
    return (exceed + 1) / (draws + 1)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(columns or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _display(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4f}"
    return str(value)


def _latex_escape(value: Any) -> str:
    text = _display(value)
    for original, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(original, replacement)
    return text


def write_table_bundle(
    root: Path,
    name: str,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str] | None = None,
    *,
    caption: str,
) -> None:
    fields = list(columns or (rows[0].keys() if rows else []))
    write_csv(root / f"{name}.csv", rows, fields)
    markdown = [f"# {caption}", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    markdown.extend("| " + " | ".join(_display(row.get(field, "")) for field in fields) + " |" for row in rows)
    (root / f"{name}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    alignment = "l" + "r" * max(0, len(fields) - 1)
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(_latex_escape(field) for field in fields) + r" \\",
        r"\midrule",
    ]
    latex.extend(" & ".join(_latex_escape(row.get(field, "")) for field in fields) + r" \\" for row in rows)
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (root / f"{name}.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def _tex_label(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _ps_escape(value: str) -> str:
    return value.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _write_bar_postscript(
    path: Path,
    *,
    title: str,
    ylabel: str,
    labels: Sequence[str],
    values: Sequence[float],
    ymax: float,
) -> None:
    left, bottom, width, height = 78.0, 72.0, 594.0, 292.0
    bar_slot = width / max(1, len(values))
    bar_width = min(46.0, bar_slot * 0.60)
    commands = [
        "%!PS-Adobe-3.0 EPSF-3.0", "%%BoundingBox: 0 0 720 432", "%%Pages: 1", "%%EndComments",
        "/Helvetica findfont 10 scalefont setfont", "1 setlinejoin", "1 setlinecap",
        f"/Helvetica-Bold findfont 15 scalefont setfont 360 404 moveto ({_ps_escape(title)}) dup stringwidth pop -2 div 0 rmoveto show",
        "/Helvetica findfont 9 scalefont setfont",
    ]
    for tick in range(5):
        fraction = tick / 4.0
        y = bottom + fraction * height
        value = fraction * ymax
        commands.extend(
            [
                "0.88 setgray 0.5 setlinewidth",
                f"{left:.2f} {y:.2f} moveto {left + width:.2f} {y:.2f} lineto stroke",
                "0 setgray",
                f"{left - 8:.2f} {y - 3:.2f} moveto ({value:.2f}) dup stringwidth pop neg 0 rmoveto show",
            ]
        )
    commands.extend(
        [
            "0 setgray 1 setlinewidth",
            f"{left:.2f} {bottom:.2f} moveto {left:.2f} {bottom + height:.2f} lineto stroke",
            f"{left:.2f} {bottom:.2f} moveto {left + width:.2f} {bottom:.2f} lineto stroke",
            f"gsave 20 {bottom + height / 2:.2f} translate 90 rotate 0 0 moveto ({_ps_escape(ylabel)}) dup stringwidth pop -2 div 0 rmoveto show grestore",
        ]
    )
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * bar_slot + (bar_slot - bar_width) / 2.0
        bar_height = 0.0 if ymax <= 0 else max(0.0, value / ymax * height)
        center = x + bar_width / 2.0
        commands.extend(
            [
                "0.26 0.47 0.75 setrgbcolor",
                f"newpath {x:.2f} {bottom:.2f} moveto {bar_width:.2f} 0 rlineto 0 {bar_height:.2f} rlineto {-bar_width:.2f} 0 rlineto closepath fill",
                "0 setgray /Helvetica findfont 8 scalefont setfont",
                f"gsave {center:.2f} {bottom - 12:.2f} translate 25 rotate 0 0 moveto ({_ps_escape(label)}) show grestore",
                f"/Helvetica findfont 8 scalefont setfont {center:.2f} {bottom + bar_height + 6:.2f} moveto ({value:.3f}) dup stringwidth pop -2 div 0 rmoveto show",
            ]
        )
    commands.extend(["showpage", "%%EOF"])
    path.write_text("\n".join(commands) + "\n", encoding="ascii", errors="replace")


def _write_line_postscript(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
    unit_square: bool,
) -> None:
    left, bottom, width, height = 82.0, 64.0, 580.0, 300.0
    all_points = [point for _, points in series for point in points]
    x_values = [point[0] for point in all_points] or [0.0, 1.0]
    y_values = [point[1] for point in all_points] or [0.0, 1.0]
    xmin, xmax = (0.0, 1.0) if unit_square else (min(x_values), max(x_values))
    ymin, ymax = (0.0, 1.0) if unit_square else (0.0, max(1e-9, max(y_values) * 1.10))
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (
            left + (x - xmin) / (xmax - xmin) * width,
            bottom + (y - ymin) / (ymax - ymin) * height,
        )

    colors = ((0.10, 0.35, 0.75), (0.78, 0.20, 0.18), (0.05, 0.55, 0.45), (0.85, 0.45, 0.05), (0.50, 0.20, 0.65))
    commands = [
        "%!PS-Adobe-3.0 EPSF-3.0", "%%BoundingBox: 0 0 720 432", "%%Pages: 1", "%%EndComments",
        "/Helvetica findfont 9 scalefont setfont", "1 setlinejoin", "1 setlinecap",
        f"/Helvetica-Bold findfont 15 scalefont setfont 360 404 moveto ({_ps_escape(title)}) dup stringwidth pop -2 div 0 rmoveto show",
        "/Helvetica findfont 9 scalefont setfont",
    ]
    for tick in range(5):
        fraction = tick / 4.0
        x = left + fraction * width
        y = bottom + fraction * height
        x_value = xmin + fraction * (xmax - xmin)
        y_value = ymin + fraction * (ymax - ymin)
        commands.extend(
            [
                "0.88 setgray 0.5 setlinewidth",
                f"{left:.2f} {y:.2f} moveto {left + width:.2f} {y:.2f} lineto stroke",
                f"{x:.2f} {bottom:.2f} moveto {x:.2f} {bottom + height:.2f} lineto stroke",
                "0 setgray",
                f"{left - 8:.2f} {y - 3:.2f} moveto ({y_value:.2f}) dup stringwidth pop neg 0 rmoveto show",
                f"{x:.2f} {bottom - 16:.2f} moveto ({x_value:.2f}) dup stringwidth pop -2 div 0 rmoveto show",
            ]
        )
    commands.extend(
        [
            "0 setgray 1 setlinewidth",
            f"{left:.2f} {bottom:.2f} moveto {left:.2f} {bottom + height:.2f} lineto stroke",
            f"{left:.2f} {bottom:.2f} moveto {left + width:.2f} {bottom:.2f} lineto stroke",
            f"{left + width / 2:.2f} 20 moveto ({_ps_escape(xlabel)}) dup stringwidth pop -2 div 0 rmoveto show",
            f"gsave 20 {bottom + height / 2:.2f} translate 90 rotate 0 0 moveto ({_ps_escape(ylabel)}) dup stringwidth pop -2 div 0 rmoveto show grestore",
        ]
    )
    if unit_square:
        x0, y0 = project((0.0, 0.0))
        x1, y1 = project((1.0, 1.0))
        commands.extend(["0.55 setgray [4 4] 0 setdash", f"{x0:.2f} {y0:.2f} moveto {x1:.2f} {y1:.2f} lineto stroke", "[] 0 setdash"])
    for index, (label, points) in enumerate(series):
        color = colors[index % len(colors)]
        commands.append(f"{color[0]} {color[1]} {color[2]} setrgbcolor 1.8 setlinewidth")
        projected = [project(point) for point in points]
        if projected:
            commands.append(f"newpath {projected[0][0]:.2f} {projected[0][1]:.2f} moveto")
            commands.extend(f"{x:.2f} {y:.2f} lineto" for x, y in projected[1:])
            commands.append("stroke")
            for x, y in projected:
                commands.append(f"newpath {x:.2f} {y:.2f} 2.7 0 360 arc closepath fill")
        legend_y = bottom + height - 14.0 * index - 10.0
        commands.extend(
            [
                f"{left + 10:.2f} {legend_y:.2f} moveto {left + 28:.2f} {legend_y:.2f} lineto stroke",
                f"0 setgray {left + 34:.2f} {legend_y - 3:.2f} moveto ({_ps_escape(label)}) show",
            ]
        )
    commands.extend(["showpage", "%%EOF"])
    path.write_text("\n".join(commands) + "\n", encoding="ascii", errors="replace")


def write_bar_figure(
    path: Path,
    *,
    title: str,
    ylabel: str,
    labels: Sequence[str],
    values: Sequence[float],
    ymax: float | None = None,
) -> None:
    symbolic = ",".join(_tex_label(label) for label in labels)
    coordinates = " ".join(f"({_tex_label(label)},{value:.8f})" for label, value in zip(labels, values))
    upper = ymax if ymax is not None else max(values, default=1.0) * 1.15
    document = rf"""\documentclass[tikz,border=4pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
  width=13cm,height=7cm,ybar,bar width=13pt,
  ymin=0,ymax={upper:.8f},ylabel={{{_tex_label(ylabel)}}},
  symbolic x coords={{{symbolic}}},xtick=data,
  x tick label style={{rotate=25,anchor=east,font=\scriptsize}},
  title={{{_tex_label(title)}}},nodes near coords,nodes near coords style={{font=\scriptsize}},
  grid=major]
\addplot[fill=blue!55,draw=blue!75!black] coordinates {{{coordinates}}};
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    path.write_text(document, encoding="utf-8")
    _write_bar_postscript(
        path.with_suffix(".ps"),
        title=title,
        ylabel=ylabel,
        labels=labels,
        values=values,
        ymax=upper,
    )


def write_line_figure(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
    unit_square: bool = False,
) -> None:
    additions = []
    legends = []
    colors = ["blue", "red", "teal", "orange", "purple", "black"]
    for index, (label, points) in enumerate(series):
        coordinates = " ".join(f"({x:.8f},{y:.8f})" for x, y in points)
        additions.append(
            rf"\addplot+[mark=*,thick,color={colors[index % len(colors)]}] coordinates {{{coordinates}}};"
        )
        legends.append(_tex_label(label))
    bounds = "xmin=0,xmax=1,ymin=0,ymax=1," if unit_square else "ymin=0,"
    diagonal = r"\addplot[dashed,gray,domain=0:1] {x};" if unit_square else ""
    document = rf"""\documentclass[tikz,border=4pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
  width=13cm,height=7cm,{bounds}
  xlabel={{{_tex_label(xlabel)}}},ylabel={{{_tex_label(ylabel)}}},
  title={{{_tex_label(title)}}},grid=major,
  legend style={{font=\scriptsize,at={{(0.02,0.98)}},anchor=north west}}]
{diagonal}
{chr(10).join(additions)}
\legend{{{','.join(legends)}}}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    path.write_text(document, encoding="utf-8")
    _write_line_postscript(
        path.with_suffix(".ps"),
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        series=series,
        unit_square=unit_square,
    )


def compile_figures(directory: Path) -> list[dict[str, Any]]:
    pdflatex = shutil.which("pdflatex") or "/home/lxc/anaconda3/bin/pdflatex"
    pdftoppm = shutil.which("pdftoppm")
    ghostscript = shutil.which("gs")
    results: list[dict[str, Any]] = []
    for tex_path in sorted(directory.glob("*.tex")):
        pdf_path = tex_path.with_suffix(".pdf")
        ps_path = tex_path.with_suffix(".ps")
        if ghostscript and ps_path.is_file():
            command = [
                ghostscript,
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-dEPSCrop",
                "-sDEVICE=pdfwrite",
                f"-sOutputFile={pdf_path}",
                str(ps_path),
            ]
        else:
            command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(directory), str(tex_path)]
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        row: dict[str, Any] = {"figure": tex_path.stem, "pdf_exit_code": run.returncode}
        if run.returncode == 0 and pdftoppm and pdf_path.is_file():
            png = subprocess.run(
                [pdftoppm, "-png", "-r", "300", "-singlefile", str(pdf_path), str(directory / tex_path.stem)],
                capture_output=True,
                text=True,
                check=False,
            )
            row["png_exit_code"] = png.returncode
        else:
            row["png_exit_code"] = None
            row["stderr_tail"] = (run.stdout + run.stderr)[-1000:]
        results.append(row)
    return results

