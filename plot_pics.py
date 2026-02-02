#!/usr/bin/env python3
"""
Plot benchmark CSV results into "pics" charts (stacked horizontal bars).

This script is designed to match the existing image style in `2022-12/pics/`:
stacked barh per runtime, per-category charts, plus `global.png/global2.png`.

Examples:
  ./venv/bin/python plot_pics.py --results-dir 2026-01/results --output-dir 2026-01/pics
  ./venv/bin/python plot_pics.py --results-dir 2022-12/res --output-dir 2026-01/pics
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimeUnit:
    label: str
    seconds: float


TIME_UNITS: list[TimeUnit] = [
    TimeUnit("h", 3600.0),
    TimeUnit("min", 60.0),
    TimeUnit("s", 1.0),
    TimeUnit("ms", 1e-3),
    TimeUnit("µs", 1e-6),
    TimeUnit("ns", 1e-9),
]


RUNTIME_DISPLAY_NAME: dict[str, str] = {
    "iwasm": "iWasm",
    "wasm2c": "Wasm2c",
    "wasmedge": "Wasmedge",
    "wasm3": "Wasm3",
    "wasmtime": "Wasmtime",
    "wasmer": "Wasmer",
    "wasmer-llvm": "Wasmer LLVM",
    "wasmer-cranelift": "Wasmer Cranelift",
    "wasmer-singlepass": "Wasmer Singlepass",
    "node": "Node",
    "bun": "Bun",
    "wazero": "Wazero",
    "native": "Native",
    "wasmoon": "Wasmoon",
}


def runtime_label(runtime: str) -> str:
    return RUNTIME_DISPLAY_NAME.get(runtime, runtime.replace("_", " ").replace("-", " ").title())


def load_csv_semicolon(csv_path: Path) -> dict[str, int]:
    results: dict[str, int] = {}
    for raw_line in csv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or ";" not in line:
            continue
        name, value = line.split(";", 1)
        name = name.strip()
        try:
            time_ns = int(value.strip())
        except ValueError:
            continue
        if time_ns <= 0:
            continue
        results[name] = time_ns
    return results


@dataclass(frozen=True)
class BadFile:
    path: Path
    reason: str


def stable_color(label: str) -> tuple[float, float, float]:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    h = int.from_bytes(digest[:2], "big") / 65535.0
    s = 0.55 + (digest[2] / 255.0) * 0.25
    v = 0.75 + (digest[3] / 255.0) * 0.20
    return colorsys.hsv_to_rgb(h, s, v)


def pick_time_unit(max_seconds: float) -> TimeUnit:
    if max_seconds <= 0:
        return TimeUnit("s", 1.0)
    for unit in TIME_UNITS:
        if max_seconds >= unit.seconds:
            return unit
    return TIME_UNITS[-1]


def ns_to_unit(time_ns: int, unit: TimeUnit) -> float:
    return (time_ns / 1e9) / unit.seconds


def sum_ns_for_tests(data: dict[str, int], tests: list[str]) -> int:
    return sum(data.get(t, 0) for t in tests)


def sorted_runtimes_for_tests(
    all_data: dict[str, dict[str, int]],
    tests: list[str],
) -> list[str]:
    totals: list[tuple[str, int]] = []
    for runtime, data in all_data.items():
        total_ns = sum_ns_for_tests(data, tests)
        if total_ns > 0:
            totals.append((runtime, total_ns))
    totals.sort(key=lambda x: x[1])
    return [r for r, _ in totals]


def plot_stacked_barh(
    *,
    output_path: Path,
    all_data: dict[str, dict[str, int]],
    runtimes: list[str],
    tests: list[str],
    title: str | None,
    legend_mode: str,
    dpi: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Error: matplotlib is required. Try: ./venv/bin/python plot_pics.py ...")

    totals_ns = [sum_ns_for_tests(all_data[r], tests) for r in runtimes]
    max_seconds = max((t / 1e9) for t in totals_ns) if totals_ns else 0.0
    unit = pick_time_unit(max_seconds)

    fig_size = (10.4, 8.0)
    if legend_mode == "right":
        # Match the wide canvas style of `2022-12/pics/global.png` so long legend
        # entries don't get clipped.
        fig_size = (22.0, 10.0)
    fig, ax = plt.subplots(figsize=fig_size)

    y = list(range(len(runtimes)))
    left = [0.0 for _ in runtimes]

    # Keep deterministic order.
    tests = [t for t in tests if any(all_data[r].get(t, 0) > 0 for r in runtimes)]

    for test in tests:
        values = [ns_to_unit(all_data[r].get(test, 0), unit) for r in runtimes]
        if all(v == 0 for v in values):
            continue
        ax.barh(
            y,
            values,
            left=left,
            height=0.7,
            color=stable_color(test),
            label=test,
        )
        left = [l + v for l, v in zip(left, values)]

    ax.set_yticks(y)
    ax.set_yticklabels([runtime_label(r) for r in runtimes], fontsize=14)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=12)
    ax.invert_yaxis()

    if title:
        ax.set_title(title, fontsize=14, pad=16)

    ax.xaxis.grid(True, color="#cccccc", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["left"].set_color("black")

    # Keep plots similar to existing pics (no explicit unit label), but ensure ticks are readable.
    ax.set_xlabel(f"Time ({unit.label})", fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    if legend_mode == "top" and labels:
        max_len = max(len(label) for label in labels)
        if max_len > 24:
            ncol = 2
        elif max_len > 18:
            ncol = 3
        elif max_len > 14:
            ncol = 4
        else:
            ncol = 6
        ncol = max(1, min(ncol, len(labels)))

        rows = math.ceil(len(labels) / ncol)
        top = 0.88 - 0.06 * (rows - 1)
        if title:
            top -= 0.03
        top = max(0.60, top)

        fig.tight_layout(rect=(0, 0, 1, top))
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=ncol,
            frameon=False,
            fontsize=14,
            handlelength=1.0,
            columnspacing=1.2,
        )
    elif legend_mode == "right" and labels:
        # Keep legend outside the plot area (like 2022-12/pics/global.png).
        ncol = 2 if len(labels) <= 70 else 3
        right = 0.76 if ncol == 2 else 0.72
        fig.tight_layout(rect=(0, 0, right, 1))
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(right + 0.02, 0.99),
            ncol=ncol,
            frameon=False,
            fontsize=9,
            handlelength=1.0,
            columnspacing=1.2,
        )
    else:
        fig.tight_layout()

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_charts(results_dir: Path, output_dir: Path, dpi: int) -> None:
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No .csv files found in {results_dir}")

    all_data: dict[str, dict[str, int]] = {}
    bad_csv_files: list[BadFile] = []
    for csv_file in csv_files:
        runtime = csv_file.stem
        try:
            data = load_csv_semicolon(csv_file)
        except Exception as e:
            bad_csv_files.append(BadFile(csv_file, f"failed to read/parse: {type(e).__name__}: {e}"))
            continue
        if not data:
            bad_csv_files.append(BadFile(csv_file, "no valid results (empty file or no parsable 'name;int' lines)"))
            continue
        all_data[runtime] = data

    if not all_data:
        # Keep the failure actionable: list all CSV files that were skipped.
        if bad_csv_files:
            lines = [f"No valid results loaded from {results_dir}. Skipped CSV files:"]
            for bad in bad_csv_files:
                lines.append(f"  - {bad.path.name}: {bad.reason}")
            raise SystemExit("\n".join(lines))
        raise SystemExit(f"No valid results loaded from {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_tests = sorted({t for data in all_data.values() for t in data.keys()})
    runtimes_global = sorted_runtimes_for_tests(all_data, all_tests)

    # If some benchmark runs produced empty/bad CSVs (e.g. test failures), make them visible.
    if bad_csv_files:
        print("Warning: skipped CSV files:", file=sys.stderr)
        for bad in bad_csv_files:
            print(f"  - {bad.path.name}: {bad.reason}", file=sys.stderr)

    # Global charts.
    bad_outputs: list[BadFile] = []
    for output_name, legend_mode in [("global2.png", "none"), ("global.png", "right")]:
        out_path = output_dir / output_name
        try:
            plot_stacked_barh(
                output_path=out_path,
                all_data=all_data,
                runtimes=runtimes_global,
                tests=all_tests,
                title=None,
                legend_mode=legend_mode,
                dpi=dpi,
            )
        except Exception as e:
            bad_outputs.append(BadFile(out_path, f"failed to plot: {type(e).__name__}: {e}"))

    # Category charts (match the existing 2022-12 picture set).
    categories: dict[str, list[str]] = {
        "aead": [t for t in all_tests if t.startswith("aead_")],
        "auth": [t for t in all_tests if t.startswith("auth")],
        "box": [t for t in all_tests if t.startswith("box")],
        "dh": [t for t in all_tests if t.startswith("scalarmult")],
        "ed25519": [t for t in all_tests if t.startswith(("core_ed25519", "core_ristretto255", "ed25519_convert"))],
        "hash": [t for t in ["generichash", "generichash2", "generichash3", "hash"] if t in all_tests],
        "keygen": [t for t in ["randombytes", "keygen"] if t in all_tests],
        "kx": [t for t in ["kdf", "kdf_hkdf", "kx"] if t in all_tests],
        "metamorphic": [t for t in ["metamorphic"] if t in all_tests],
        "onetimeauth": [t for t in ["onetimeauth7"] if t in all_tests],
        "pwhash": [t for t in ["pwhash_argon2i", "pwhash_argon2id", "pwhash_scrypt", "pwhash_scrypt_ll"] if t in all_tests],
        "secretbox": [t for t in ["secretbox7", "secretbox8", "secretbox_easy", "secretbox_easy2", "secretstream_xchacha20poly1305"] if t in all_tests],
        "sign": [t for t in ["sign", "sign2"] if t in all_tests],
        "stream": [t for t in ["stream", "stream2", "chacha20", "xchacha20"] if t in all_tests],
        "utils": [t for t in ["core3", "codecs", "sodium_utils", "verify1"] if t in all_tests],
    }

    for category, tests in categories.items():
        if not tests:
            continue
        runtimes = sorted_runtimes_for_tests(all_data, tests)
        if not runtimes:
            continue
        out_path = output_dir / f"{category}.png"
        try:
            plot_stacked_barh(
                output_path=out_path,
                all_data=all_data,
                runtimes=runtimes,
                tests=tests,
                title=None,
                legend_mode="top",
                dpi=dpi,
            )
        except Exception as e:
            bad_outputs.append(BadFile(out_path, f"failed to plot: {type(e).__name__}: {e}"))

    if bad_outputs:
        print("Error: failed to generate some charts:", file=sys.stderr)
        for bad in bad_outputs:
            print(f"  - {bad.path.name}: {bad.reason}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark CSV results to pics (stacked barh charts).")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory containing *.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("pics"), help="Directory to write *.png charts")
    parser.add_argument("--dpi", type=int, default=120, help="PNG DPI (default: 120)")

    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()

    build_charts(results_dir, output_dir, dpi=args.dpi)
    print(f"Charts saved to {output_dir}")


if __name__ == "__main__":
    main()
