# Performance Characteristics

## Complexity by phase

| Phase | Complexity | Notes |
|---|---|---|
| Distance matrix | O(n²) | n = number of points; trivially parallelised by EXLA |
| Filtration construction | O(m) | m = total simplices = Σ C(n,k) for k = 0..max_dim |
| Boundary matrix | O(m) | Sparse representation; one MapSet per non-vertex simplex |
| Reduction | O(m³) worst-case | Apparent pairs optimisation reduces this significantly in practice |

The number of simplices m grows as O(nᵈ) where d = max_dim, so the dominant cost shifts from the distance matrix to reduction as n increases.

## Practical limits

| Point cloud size | Recommended setup |
|---|---|
| n < 100 | CPU only (BinaryBackend), any max_dim |
| 100 ≤ n < 1000 | EXLA for distance matrix; reduction is the bottleneck |
| n ≥ 1000 | Use `threshold` option to limit m; EXLA strongly recommended |

## Benchmark results (reference hardware)

Measured on a 50-point cloud (`test/fixtures/o3_50.txt`, max_dim 2):

| Phase | BinaryBackend | EXLA (CPU) |
|---|---|---|
| Distance matrix | ~3 ms | ~0.05 ms (~57× faster) |
| Filtration | ~1 ms | ~1 ms |
| Boundary matrix | ~0.5 ms | ~0.5 ms |
| Reduction | ~18 ms | ~17 ms (~7% faster) |

Reduction dominates (~82% of total time) and is CPU-bound regardless of backend.
EXLA's benefit is concentrated in the distance matrix phase.

## Controlling simplex count

The `threshold` option is the primary lever for large inputs. The default (enclosing radius) already limits the filtration to the topologically meaningful range:

```elixir
# Default: threshold = enclosing radius (recommended)
PhNx.compute(points, max_dim: 2)

# Explicit threshold — ignore simplices born after ε = 1.5
PhNx.compute(points, max_dim: 2, threshold: 1.5)

# No threshold — include everything (can be very large)
PhNx.compute(points, max_dim: 2, threshold: :infinity)
```

To detect Hₖ features you need simplices up to dimension k+1, so keep max_dim as low as your use case allows.
