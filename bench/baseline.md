# ph-nx Performance Baseline

**Dataset**: `test/fixtures/o3_50.txt` — 50 points, 9 dimensions
**Settings**: `max_dim: 2`, threshold: enclosing radius (default)
**Simplices**: 16,598 (vertices + edges + triangles, after enclosing-radius threshold)
**Elixir**: 1.18.4 / OTP 27 / Linux 6.8 / RTX 3090 / CUDA 13

## Per-phase timing

| Phase            | BinaryBackend | EXLA (CPU) |
|------------------|---------------|------------|
| Distance matrix  | 51.8 ms       | 0.9 ms     |
| Filtration build | 31.7 ms       | 21.3 ms    |
| Boundary matrix  | 27.5 ms       | 21.3 ms    |
| Reduction        | 523.4 ms      | 489.5 ms   |
| **Total**        | **634.3 ms**  | **533.0 ms** |

## Multi-run averages (5 runs, after 2 warm-up)

| Backend       | Average    | Min        |
|---------------|------------|------------|
| BinaryBackend | 627.5 ms   | 602.2 ms   |
| EXLA (CPU)    | 558.2 ms   | 553.6 ms   |

## Observations

- **Reduction dominates**: 523 ms / 634 ms = **82% of total runtime** (BinaryBackend)
- **EXLA wins on distance matrix**: 0.9 ms vs 51.8 ms (~57×), but that phase is <10% of total
- **EXLA speeds up reduction**: 489 ms vs 523 ms (~7% faster) — apparent-pairs pre-pass
  benefits from EXLA's native integer ops
- **Apparent-pairs optimisation active**: benchmark now calls `Reduction.reduce/3`
  (with filtration) matching what `PhNx.compute/2` uses in production (fixed in #27)

## Running EXLA locally

EXLA's CUDA 13 binary requires NVSHMEM. On this machine it is provided by the
`nvidia-nvshmem-cu12` Python package. A `.envrc` (direnv) file at the repo root
sets `LD_LIBRARY_PATH` automatically:

```bash
direnv allow      # once, after cloning
mix run bench/benchmark.exs
```

Without direnv, set the path manually:

```bash
NVSHMEM_LIB=$(python3 -c "import nvidia.nvshmem, os; print(os.path.join(os.path.dirname(nvidia.nvshmem.__file__), 'lib'))")
LD_LIBRARY_PATH="$NVSHMEM_LIB" mix run bench/benchmark.exs
```

## Pre-optimisation reference (original baseline, issues #3–#6)

Recorded before threshold defaulting and apparent-pairs; 20,875 simplices, no threshold.

| Phase            | BinaryBackend | EXLA (CPU)   |
|------------------|---------------|--------------|
| Distance matrix  | 18.8 ms       | 0.9 ms       |
| Filtration build | 33.0 ms       | 36.3 ms      |
| Boundary matrix  | 24.3 ms       | 28.4 ms      |
| Reduction        | 547.3 ms      | 546.3 ms     |
| **Total**        | **623.4 ms**  | **611.9 ms** |

## Correctness baseline (ripser --format point-cloud --dim 1)

| Metric    | Expected | Actual |
|-----------|----------|--------|
| H₀ pairs  | 49       | 49 ✓   |
| H₁ pairs  | 51       | 51 ✓   |
| Tolerance | 1e-4     | pass ✓ |
