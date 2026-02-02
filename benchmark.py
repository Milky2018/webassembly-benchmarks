#!/usr/bin/env python3
"""
WebAssembly Benchmark Runner and Visualization Tool

This script runs WebAssembly benchmarks across multiple runtimes,
collects results to CSV files, and generates comparison charts.

Usage:
    python benchmark.py run --wasm-dir ./wasm --output-dir ./results
    python benchmark.py plot --csv-dir ./results --output-dir ./charts
    python benchmark.py all --wasm-dir ./wasm --output-dir ./results
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Local bin directory (relative to script location)
SCRIPT_DIR = Path(__file__).parent.resolve()
LOCAL_BIN_DIR = SCRIPT_DIR / "bin"

# Runtime configurations: (name, command_template)
# Use {wasm_file} as placeholder for the wasm file path
# Use {runtime_bin} as placeholder for the runtime binary path
RUNTIMES = {
    "wasmtime": "wasmtime run {wasm_file}",
    "wasmer": "wasmer run {wasm_file}",
    "wasmer-cranelift": "wasmer run --cranelift {wasm_file}",
    "wasmer-llvm": "wasmer run --llvm {wasm_file}",
    "wasmer-singlepass": "wasmer run --singlepass {wasm_file}",
    "node": "node --no-warnings -e \"const fs=require('fs');const {{WASI}}=require('wasi');let wasi;try{{wasi=new WASI();}}catch(e){{wasi=new WASI({{version:'preview1'}});}}const mod=new WebAssembly.Module(fs.readFileSync('{wasm_file}'));const inst=new WebAssembly.Instance(mod,wasi.getImportObject());wasi.start(inst);\"",
    "wasm3": "wasm3 {wasm_file}",
    "iwasm": "iwasm {wasm_file}",
    "wasmedge": "wasmedge {wasm_file}",
    "wavm": "wavm run {wasm_file}",
    "wazero": "wazero run {wasm_file}",
    "wasmoon": "wasmoon run {wasm_file}",
}


@dataclass
class BenchmarkResult:
    """Represents a single benchmark result."""
    test_name: str
    time_ns: int
    success: bool
    error: Optional[str] = None


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_progress(
    current: int,
    total: int,
    test_name: str,
    elapsed: float,
    success_count: int,
    fail_count: int,
    bar_width: int = 30,
) -> None:
    """Print a progress bar with status information."""
    percent = current / total
    filled = int(bar_width * percent)
    bar = "=" * filled + ">" + " " * (bar_width - filled - 1)

    # Estimate remaining time
    if current > 0:
        eta = elapsed / current * (total - current)
        eta_str = format_duration(eta)
    else:
        eta_str = "..."

    # Truncate test name if too long
    max_name_len = 25
    if len(test_name) > max_name_len:
        test_name = test_name[:max_name_len-2] + ".."

    status = f"\r  [{bar}] {current}/{total} | {test_name:<{max_name_len}} | OK:{success_count} FAIL:{fail_count} | ETA: {eta_str}  "
    print(status, end="", flush=True)


def find_runtime_binary(runtime: str) -> Optional[Path]:
    """Find the runtime binary, checking local bin first, then system PATH."""
    cmd_template = RUNTIMES.get(runtime, "")
    if not cmd_template:
        return None

    cmd = cmd_template.split()[0]

    # Check local bin directory first
    local_bin = LOCAL_BIN_DIR / cmd
    if local_bin.exists() and local_bin.is_file():
        return local_bin

    # Check system PATH
    try:
        result = subprocess.run(
            ["which", cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


def check_runtime_available(runtime: str) -> bool:
    """Check if a runtime is available in the system or local bin."""
    return find_runtime_binary(runtime) is not None


def list_available_runtimes() -> list[str]:
    """List all available runtimes on the system."""
    available = []
    for runtime in RUNTIMES:
        if check_runtime_available(runtime):
            available.append(runtime)
    return available


def run_single_benchmark(
    wasm_file: Path,
    runtime: str,
    timeout: int = 300,
) -> BenchmarkResult:
    """Run a single benchmark and return the result."""
    test_name = wasm_file.stem
    cmd_template = RUNTIMES.get(runtime)

    if not cmd_template:
        return BenchmarkResult(
            test_name=test_name,
            time_ns=0,
            success=False,
            error=f"Unknown runtime: {runtime}",
        )

    # Find the runtime binary (local bin or system PATH)
    runtime_bin = find_runtime_binary(runtime)
    if not runtime_bin:
        return BenchmarkResult(
            test_name=test_name,
            time_ns=0,
            success=False,
            error=f"Runtime not found: {runtime}",
        )

    # Replace the command name with the full path
    cmd_parts = cmd_template.split()
    cmd_parts[0] = str(runtime_bin)
    cmd = " ".join(cmd_parts).format(wasm_file=str(wasm_file))

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout + result.stderr

        # Parse the output - the benchmark prints time in nanoseconds
        # Format: <number>\nPASS <test_name> (exit status: 0)
        # Or just a number on the first line
        lines = output.strip().split("\n")

        for line in lines:
            line = line.strip()
            # Try to find a line that's just a number (nanoseconds)
            if line.isdigit():
                return BenchmarkResult(
                    test_name=test_name,
                    time_ns=int(line),
                    success=True,
                )

        # If no pure number found, try regex
        match = re.search(r"^(\d+)$", output, re.MULTILINE)
        if match:
            return BenchmarkResult(
                test_name=test_name,
                time_ns=int(match.group(1)),
                success=True,
            )

        return BenchmarkResult(
            test_name=test_name,
            time_ns=0,
            success=False,
            error=f"Could not parse output: {output[:200]}",
        )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            test_name=test_name,
            time_ns=0,
            success=False,
            error=f"Timeout after {timeout}s",
        )
    except Exception as e:
        return BenchmarkResult(
            test_name=test_name,
            time_ns=0,
            success=False,
            error=str(e),
        )


def run_benchmarks(
    wasm_dir: Path,
    output_dir: Path,
    runtimes: Optional[list[str]] = None,
    timeout: int = 300,
    verbose: bool = False,
) -> dict[str, list[BenchmarkResult]]:
    """Run benchmarks for all wasm files across specified runtimes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of wasm files
    wasm_files = sorted(wasm_dir.glob("*.wasm"))
    if not wasm_files:
        print(f"Error: No .wasm files found in {wasm_dir}")
        sys.exit(1)

    print(f"Found {len(wasm_files)} wasm files in {wasm_dir}")

    # Determine which runtimes to use
    if runtimes is None:
        runtimes = list_available_runtimes()

    if not runtimes:
        print("Error: No runtimes available")
        sys.exit(1)

    print(f"Using runtimes: {', '.join(runtimes)}")

    all_results: dict[str, list[BenchmarkResult]] = {}

    for runtime_idx, runtime in enumerate(runtimes, 1):
        print(f"\n{'='*60}")
        print(f"[{runtime_idx}/{len(runtimes)}] Running benchmarks with: {runtime}")
        print(f"{'='*60}")

        results: list[BenchmarkResult] = []
        success_count = 0
        fail_count = 0
        failed_files: list[tuple[str, str]] = []  # (wasm filename, error)
        start_time = time.time()

        for i, wasm_file in enumerate(wasm_files, 1):
            # Show progress before running
            print_progress(
                current=i - 1,
                total=len(wasm_files),
                test_name=wasm_file.stem,
                elapsed=time.time() - start_time,
                success_count=success_count,
                fail_count=fail_count,
            )

            result = run_single_benchmark(wasm_file, runtime, timeout)
            results.append(result)

            if result.success:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append((wasm_file.name, result.error or ""))

            # Show progress after running
            print_progress(
                current=i,
                total=len(wasm_files),
                test_name=wasm_file.stem,
                elapsed=time.time() - start_time,
                success_count=success_count,
                fail_count=fail_count,
            )

            # In verbose mode, print details on new line
            if verbose:
                if result.success:
                    print(f"\n    -> {result.time_ns} ns")
                else:
                    print(f"\n    -> FAILED: {result.error}")

        # Clear progress line and print summary
        elapsed_total = time.time() - start_time
        print(f"\r{' ' * 100}\r", end="")  # Clear line

        all_results[runtime] = results

        # Save to CSV
        csv_path = output_dir / f"{runtime}.csv"
        save_results_to_csv(results, csv_path)

        # Print summary
        print(f"  Completed in {format_duration(elapsed_total)}: {success_count}/{len(results)} passed, {fail_count} failed")
        print(f"  Results saved to {csv_path}")
        if failed_files:
            print("  Failed wasm files:")
            for fname, err in failed_files:
                if verbose:
                    err_str = err.strip()
                else:
                    # Keep the non-verbose summary scannable.
                    err_str = (err.splitlines()[0].strip() if err else "")
                if err_str:
                    print(f"    - {fname}: {err_str}")
                else:
                    print(f"    - {fname}")

    return all_results


def save_results_to_csv(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to a CSV file."""
    with open(output_path, "w", newline="") as f:
        for result in results:
            if result.success:
                f.write(f"{result.test_name};{result.time_ns}\n")


def load_csv_results(csv_path: Path) -> dict[str, int]:
    """Load benchmark results from a CSV file."""
    results = {}
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            parts = line.split(";")
            if len(parts) >= 2:
                test_name = parts[0]
                try:
                    time_ns = int(parts[1])
                    results[test_name] = time_ns
                except ValueError:
                    continue
    return results


def generate_charts(
    csv_dir: Path,
    output_dir: Path,
    chart_types: Optional[list[str]] = None,
) -> None:
    """Generate comparison charts from CSV files."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib and numpy are required for chart generation")
        print("Install with: pip install matplotlib numpy")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all CSV files
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {csv_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files")

    # Load data from all runtimes
    all_data: dict[str, dict[str, int]] = {}
    for csv_file in csv_files:
        runtime = csv_file.stem
        all_data[runtime] = load_csv_results(csv_file)
        print(f"  Loaded {len(all_data[runtime])} results from {runtime}")

    # Get all unique test names
    all_tests = set()
    for data in all_data.values():
        all_tests.update(data.keys())
    all_tests = sorted(all_tests)

    if chart_types is None:
        chart_types = ["global", "categories", "vs_native"]

    # Generate global comparison chart
    if "global" in chart_types:
        generate_global_chart(all_data, all_tests, output_dir)

    # Generate category-based charts
    if "categories" in chart_types:
        generate_category_charts(all_data, all_tests, output_dir)

    # Generate vs native comparison
    if "vs_native" in chart_types and "native" in all_data:
        generate_vs_native_chart(all_data, all_tests, output_dir)

    print(f"\nCharts saved to {output_dir}")


def generate_global_chart(
    all_data: dict[str, dict[str, int]],
    all_tests: list[str],
    output_dir: Path,
) -> None:
    """Generate a global comparison chart showing total execution time."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate total time for each runtime
    totals = {}
    for runtime, data in all_data.items():
        total = sum(data.values())
        totals[runtime] = total

    # Sort by total time
    sorted_runtimes = sorted(totals.keys(), key=lambda x: totals[x])

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_runtimes)))
    bars = ax.barh(sorted_runtimes, [totals[r] / 1e9 for r in sorted_runtimes], color=colors)

    ax.set_xlabel("Total Execution Time (seconds)")
    ax.set_title("WebAssembly Runtime Benchmark - Total Execution Time")
    ax.set_xlim(0, max(totals.values()) / 1e9 * 1.1)

    # Add value labels
    for bar, runtime in zip(bars, sorted_runtimes):
        width = bar.get_width()
        ax.text(
            width + max(totals.values()) / 1e9 * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}s",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "global.png", dpi=150)
    plt.close()
    print("  Generated global.png")


def generate_category_charts(
    all_data: dict[str, dict[str, int]],
    all_tests: list[str],
    output_dir: Path,
) -> None:
    """Generate charts for each category of tests."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Define categories based on test name prefixes
    categories = {
        "aead": ["aead_"],
        "auth": ["auth"],
        "box": ["box"],
        "hash": ["hash", "generichash", "shorthash"],
        "pwhash": ["pwhash"],
        "sign": ["sign"],
        "stream": ["stream", "chacha20", "xchacha20"],
        "scalarmult": ["scalarmult"],
        "secretbox": ["secretbox"],
        "kdf": ["kdf"],
        "ed25519": ["ed25519", "core_ed25519"],
        "ristretto": ["ristretto"],
    }

    for category, prefixes in categories.items():
        # Find tests matching this category
        matching_tests = []
        for test in all_tests:
            for prefix in prefixes:
                if test.startswith(prefix):
                    matching_tests.append(test)
                    break

        if not matching_tests:
            continue

        # Get runtimes that have data for these tests
        runtimes = [r for r in all_data.keys() if any(t in all_data[r] for t in matching_tests)]
        if not runtimes:
            continue

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(matching_tests))
        width = 0.8 / len(runtimes)
        colors = plt.cm.Set2(np.linspace(0, 1, len(runtimes)))

        for i, runtime in enumerate(runtimes):
            values = []
            for test in matching_tests:
                val = all_data[runtime].get(test, 0)
                values.append(val / 1e6)  # Convert to milliseconds

            offset = (i - len(runtimes) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=runtime, color=colors[i])

        ax.set_xlabel("Test")
        ax.set_ylabel("Execution Time (ms)")
        ax.set_title(f"WebAssembly Benchmark - {category.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(matching_tests, rotation=45, ha="right")
        ax.legend(loc="upper right")
        ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(output_dir / f"{category}.png", dpi=150)
        plt.close()
        print(f"  Generated {category}.png")


def generate_vs_native_chart(
    all_data: dict[str, dict[str, int]],
    all_tests: list[str],
    output_dir: Path,
) -> None:
    """Generate a chart comparing runtimes against native performance."""
    import matplotlib.pyplot as plt
    import numpy as np

    native_data = all_data.get("native", {})
    if not native_data:
        print("  Skipping vs_native chart: no native data")
        return

    # Calculate slowdown ratios for each runtime
    runtimes = [r for r in all_data.keys() if r != "native"]
    ratios = {}

    for runtime in runtimes:
        runtime_ratios = []
        for test in all_tests:
            if test in native_data and test in all_data[runtime]:
                native_time = native_data[test]
                runtime_time = all_data[runtime][test]
                if native_time > 0:
                    ratio = runtime_time / native_time
                    runtime_ratios.append(ratio)
        if runtime_ratios:
            # Use geometric mean for ratios
            ratios[runtime] = np.exp(np.mean(np.log(runtime_ratios)))

    if not ratios:
        print("  Skipping vs_native chart: no comparable data")
        return

    # Sort by ratio (fastest first)
    sorted_runtimes = sorted(ratios.keys(), key=lambda x: ratios[x])

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_runtimes)))
    bars = ax.barh(sorted_runtimes, [ratios[r] for r in sorted_runtimes], color=colors)

    ax.axvline(x=1, color="black", linestyle="--", linewidth=1, label="Native (1x)")
    ax.set_xlabel("Slowdown vs Native (lower is better)")
    ax.set_title("WebAssembly Runtime Performance vs Native Code")

    # Add value labels
    for bar, runtime in zip(bars, sorted_runtimes):
        width = bar.get_width()
        ax.text(
            width + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}x",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "vs_native.png", dpi=150)
    plt.close()
    print("  Generated vs_native.png")


def main():
    parser = argparse.ArgumentParser(
        description="WebAssembly Benchmark Runner and Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmarks with all available runtimes
  python benchmark.py run --wasm-dir ./2022-12/wasm --output-dir ./results

  # Run benchmarks with specific runtimes
  python benchmark.py run --wasm-dir ./2022-12/wasm --output-dir ./results --runtimes wasmtime wasmer

  # Generate charts from existing CSV files
  python benchmark.py plot --csv-dir ./2022-12/res --output-dir ./charts

  # Run benchmarks and generate charts
  python benchmark.py all --wasm-dir ./2022-12/wasm --output-dir ./results

  # List available runtimes
  python benchmark.py list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--wasm-dir",
        type=Path,
        required=True,
        help="Directory containing .wasm files",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Directory to save CSV results (default: ./results)",
    )
    run_parser.add_argument(
        "--runtimes",
        nargs="+",
        help="Specific runtimes to use (default: all available)",
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each test (default: 300)",
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # Plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Generate charts from CSV files")
    plot_parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Directory containing CSV files",
    )
    plot_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./charts"),
        help="Directory to save charts (default: ./charts)",
    )
    plot_parser.add_argument(
        "--charts",
        nargs="+",
        choices=["global", "categories", "vs_native"],
        help="Types of charts to generate (default: all)",
    )

    # All subcommand
    all_parser = subparsers.add_parser("all", help="Run benchmarks and generate charts")
    all_parser.add_argument(
        "--wasm-dir",
        type=Path,
        required=True,
        help="Directory containing .wasm files",
    )
    all_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Directory to save results and charts (default: ./results)",
    )
    all_parser.add_argument(
        "--runtimes",
        nargs="+",
        help="Specific runtimes to use (default: all available)",
    )
    all_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each test (default: 300)",
    )
    all_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List available runtimes")

    args = parser.parse_args()

    if args.command == "run":
        run_benchmarks(
            wasm_dir=args.wasm_dir,
            output_dir=args.output_dir,
            runtimes=args.runtimes,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    elif args.command == "plot":
        generate_charts(
            csv_dir=args.csv_dir,
            output_dir=args.output_dir,
            chart_types=args.charts,
        )

    elif args.command == "all":
        run_benchmarks(
            wasm_dir=args.wasm_dir,
            output_dir=args.output_dir,
            runtimes=args.runtimes,
            timeout=args.timeout,
            verbose=args.verbose,
        )
        generate_charts(
            csv_dir=args.output_dir,
            output_dir=args.output_dir / "charts",
        )

    elif args.command == "list":
        print(f"Local bin directory: {LOCAL_BIN_DIR}")
        print(f"Configured runtimes:")
        for runtime in RUNTIMES:
            bin_path = find_runtime_binary(runtime)
            if bin_path:
                # Check if it's from local bin or system PATH
                if str(bin_path).startswith(str(LOCAL_BIN_DIR)):
                    location = "local bin"
                else:
                    location = "system"
                print(f"  {runtime}: {bin_path} ({location})")
            else:
                print(f"  {runtime}: not found")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
