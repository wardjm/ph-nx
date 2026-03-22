# ph-nx Performance Baseline

**Dataset**: `test/fixtures/o3_50.txt` — 50 points, 9 dimensions
**Settings**: `max_dim: 2`, threshold: enclosing radius (default)
**Elixir**: 1.18.4 / OTP 27 / Linux 6.8

## Per-phase timing (BinaryBackend)

| Phase            | Time      |
|------------------|-----------|
| Distance matrix  | 57.5 ms   |
| Filtration build | 24.7 ms   |
| Boundary matrix  | 26.4 ms   |
| Reduction        | 519.3 ms  |
| **Total**        | **627.9 ms** |

## Multi-run averages (5 runs, after 2 warm-up)

| Backend       | Average    | Min        |
|---------------|------------|------------|
| BinaryBackend | 627.4 ms   | 610.0 ms   |
| EXLA (CPU)    | —          | —          |

EXLA timings not recorded: XLA binaries unavailable in this environment.

## Observations

- **Simplices**: 16,598 — lower than the original baseline (20,875) because the
  enclosing-radius threshold is now applied by default (added in PR #3)
- **Reduction dominates**: 519 ms / 628 ms = **83% of total runtime**
- **Apparent-pairs optimisation now active in benchmark**: phase breakdown previously
  called `Reduction.reduce/2` (bypassing apparent-pairs); corrected to `reduce/3`
  in this baseline (closes #27)

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
