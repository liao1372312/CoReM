#!/usr/bin/env python3
"""Plot measured held-out and transition results for public workflows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

DISPLAY = {
    "corem_reference": "CoReM reference",
    "version_bm25": "Version BM25",
    "recency_bm25": "Recency BM25",
    "service_bm25": "Service BM25",
    "no_memory": "No memory",
}
COLORS = {
    "corem_reference": "#3568b8",
    "version_bm25": "#2a9d8f",
    "recency_bm25": "#7a5195",
    "service_bm25": "#e76f51",
    "no_memory": "#777777",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_main(metrics_path: Path, output_root: Path) -> None:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        by_method = {row["method"]: row for row in csv.DictReader(handle)}
    methods = [
        "corem_reference",
        "version_bm25",
        "recency_bm25",
        "service_bm25",
        "no_memory",
    ]
    values = [float(by_method[method]["vsr"]) * 100 for method in methods]
    lows = [float(by_method[method]["vsr_ci_low"]) * 100 for method in methods]
    highs = [float(by_method[method]["vsr_ci_high"]) * 100 for method in methods]
    errors = [
        [value - low for value, low in zip(values, lows, strict=True)],
        [high - value for value, high in zip(values, highs, strict=True)],
    ]
    figure, axis = plt.subplots(figsize=(4.8, 2.65))
    positions = list(range(len(methods)))
    axis.barh(
        positions,
        values,
        xerr=errors,
        color=[COLORS[method] for method in methods],
        capsize=3,
        height=0.62,
    )
    axis.set_yticks(positions, [DISPLAY[method] for method in methods])
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xlabel("Verified success rate (%)")
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    for position, value in zip(positions, values, strict=True):
        axis.text(
            min(101, value + 1.0), position, f"{value:.1f}", va="center", fontsize=8
        )
    figure.tight_layout()
    figure.savefig(output_root / "corem_main_bar.pdf", bbox_inches="tight")
    figure.savefig(output_root / "corem_main_bar.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def transition_windows(
    scored_path: Path, events_path: Path, transitions_path: Path
) -> list[dict[str, Any]]:
    scored = read_jsonl(scored_path)
    events = {row["event_id"]: row for row in read_jsonl(events_path)}
    transitions = read_jsonl(transitions_path)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[(str(row["method"]), int(row["seed"]))].append(row)
    values: dict[tuple[str, int], list[int]] = defaultdict(list)
    for transition in transitions:
        transition_time = transition["transition_time"]
        for (method, seed), rows in grouped.items():
            relevant = [
                row
                for row in rows
                if events[row["event_id"]]["service_id"] == transition["service_id"]
                and events[row["event_id"]]["provider_id"] == transition["provider_id"]
            ]
            pre = sorted(
                (
                    row
                    for row in relevant
                    if events[row["event_id"]]["version_id"]
                    == transition["from_version_id"]
                    and events[row["event_id"]]["event_time"] < transition_time
                ),
                key=lambda row: events[row["event_id"]]["event_time"],
            )
            post = sorted(
                (
                    row
                    for row in relevant
                    if events[row["event_id"]]["version_id"]
                    == transition["to_version_id"]
                    and events[row["event_id"]]["event_time"] >= transition_time
                ),
                key=lambda row: events[row["event_id"]]["event_time"],
            )
            if pre:
                values[(method, -1)].append(int(pre[-1]["verified_success"]))
            for attempt, row in enumerate(post, start=1):
                values[(method, attempt)].append(int(row["verified_success"]))
    output = []
    for (method, attempt), outcomes in sorted(values.items()):
        output.append(
            {
                "method": method,
                "attempt": attempt,
                "observations": len(outcomes),
                "vsr": sum(outcomes) / len(outcomes),
            }
        )
    return output


def plot_drift(rows: list[dict[str, Any]], output_root: Path) -> None:
    figure, axis = plt.subplots(figsize=(4.8, 2.65))
    methods = [
        "service_bm25",
        "recency_bm25",
        "version_bm25",
        "corem_reference",
    ]
    styles = {
        "service_bm25": ("-", "o"),
        "recency_bm25": ("--", "s"),
        "version_bm25": ((0, (3, 1, 1, 1)), "^"),
        "corem_reference": ("-", "D"),
    }
    for method in methods:
        members = [row for row in rows if row["method"] == method]
        x = [int(row["attempt"]) for row in members]
        y = [float(row["vsr"]) * 100 for row in members]
        line_style, marker = styles[method]
        axis.plot(
            x,
            y,
            label=DISPLAY[method],
            color=COLORS[method],
            linestyle=line_style,
            marker=marker,
            markersize=3,
            linewidth=1.3,
        )
    axis.axvline(0, color="#444444", linestyle=":", linewidth=1)
    axis.text(0.3, 51, "release transition", fontsize=7, rotation=90, va="bottom")
    axis.set_xlim(-1.5, 20.5)
    axis.set_ylim(48, 102)
    axis.set_xlabel("Attempt relative to release transition")
    axis.set_ylabel("VSR (%)")
    axis.grid(color="#dddddd", linewidth=0.6)
    axis.legend(fontsize=6.8, loc="lower right", ncol=2)
    figure.tight_layout()
    figure.savefig(output_root / "corem_drift_windows.pdf", bbox_inches="tight")
    figure.savefig(
        output_root / "corem_drift_windows.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test-analysis", type=Path, required=True)
    parser.add_argument("--longitudinal-analysis", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_main(args.test_analysis / "method_metrics.csv", args.figures)
    windows = transition_windows(
        args.longitudinal_analysis / "scored_executions.jsonl",
        args.dataset / "events.jsonl",
        args.dataset / "version_transitions.jsonl",
    )
    window_path = args.longitudinal_analysis / "post_transition_windows.csv"
    with window_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("method", "attempt", "observations", "vsr")
        )
        writer.writeheader()
        writer.writerows(windows)
    checksum_path = args.longitudinal_analysis / "checksums.sha256"
    existing = [
        line
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if not line.endswith("  post_transition_windows.csv")
    ]
    existing.append(
        f"{hashlib.sha256(window_path.read_bytes()).hexdigest()}  "
        "post_transition_windows.csv"
    )
    checksum_path.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")
    plot_drift(windows, args.figures)
    print(f"wrote two PDF/PNG figure pairs and {window_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
