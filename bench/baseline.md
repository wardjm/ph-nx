# ph-nx Performance Baseline

Recorded before any performance optimisations (issues #3–#6).

**Dataset**: `test/fixtures/o3_50.txt` — 50 points, 9 dimensions
**Settings**: `max_dim: 2`
**Simplices**: 20,875 (vertices + edges + triangles, no threshold)
**Elixir**: 1.19.5 / OTP 27 / WSL2 x86-64

## Per-phase timing

| Phase            | BinaryBackend | EXLA (CPU) |
|------------------|---------------|------------|
| Distance matrix  | 18.8 ms       | 0.9 ms     |
| Filtration build | 33.0 ms       | 36.3 ms    |
| Boundary matrix  | 24.3 ms       | 28.4 ms    |
| Reduction        | 547.3 ms      | 546.3 ms   |
| **Total**        | **623.4 ms**  | **611.9 ms** |

## Multi-run averages (5 runs, after 2 warm-up)

| Backend       | Average    | Min        |
|---------------|------------|------------|
| BinaryBackend | 641.7 ms   | 631.8 ms   |
| EXLA (CPU)    | 604.8 ms   | 576.8 ms   |

## Observations

- **Reduction dominates**: 547 ms / 623 ms = **88% of total runtime**
- **EXLA wins on distance matrix only**: 0.9 ms vs 18.8 ms (~21×), but that phase is only 3% of total
- **No threshold applied**: all C(50,3) = 19,600 triangles are included regardless of birth time; this is the primary driver of simplex count

## Correctness baseline (ripser --format point-cloud --dim 1)

| Metric    | Expected | Actual |
|-----------|----------|--------|
| H₀ pairs  | 49       | 49 ✓   |
| H₁ pairs  | 51       | 51 ✓   |
| Tolerance | 1e-4     | pass ✓ |
