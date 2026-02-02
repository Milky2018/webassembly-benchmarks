# Benchmarks

```shell
python benchmark.py run --wasm-dir ./2022-12/wasm --output-dir ./2026-01/results --runtimes wasmtime wasmer wasmoon
python plot_pics.py --results-dir 2026-01/results --output-dir 2026-01/pics
```