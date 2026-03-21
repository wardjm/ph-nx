# ph-nx

Persistent homology in Elixir using [Nx](https://github.com/elixir-nx/nx).

Computes the persistent homology of a finite point cloud via the **Vietoris-Rips filtration** and the **standard persistence algorithm over F₂** (Edelsbrunner, Letscher, Zomorodian 2002). Results match [ripser](https://github.com/Ripser/ripser) exactly.

## Quick start

```elixir
# Four corners of a unit square — expect H₀: 1 component, H₁: 1 loop
points = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
result = PhNx.compute(points, max_dim: 2)
PhNx.print_barcode(result)
```

```
Persistence Barcode
──────────────────────────────────────────────────

H0 (4 bars):
  [0.0000, 1.0000)  persistence: 1.0000
  [0.0000, 1.0000)  persistence: 1.0000
  [0.0000, 1.0000)  persistence: 1.0000
  [0.0000, ∞)

H1 (1 bar):
  [1.0000, 1.4142)  persistence: 0.4142
```

## Installation

```elixir
def deps do
  [
    {:ph_nx, github: "wardjm/ph-nx"}
  ]
end
```

Optionally add [EXLA](https://github.com/elixir-nx/nx/tree/main/exla) for accelerated distance matrix computation:

```elixir
{:exla, "~> 0.11.0"}
```

And set it as the default backend in `config/config.exs`:

```elixir
config :nx, default_backend: EXLA.Backend
config :exla,
  preferred_clients: [:default],
  clients: [default: [platform: :host]]
```

## API

```elixir
# Compute persistent homology
result = PhNx.compute(points, max_dim: 2)
# => %{pairs: [{dim, birth, death}], essential: [{dim, birth}], diagram: [...]}

# Print barcode
PhNx.print_barcode(result)

# Betti numbers (count of essential/infinite bars per dimension)
PhNx.betti_numbers(result)
# => %{0 => 1, 1 => 0}

# Top features by persistence
PhNx.most_persistent(result, 5)
# => [{dim, birth, death, persistence}, ...]
```

### Options for `compute/2`

| Option | Default | Description |
|---|---|---|
| `:max_dim` | `2` | Maximum simplex dimension. To detect Hₖ features you need simplices up to dimension k+1. |
| `:threshold` | `:infinity` | Ignore simplices born after this filtration value. |

## How it works

### Pipeline

```
Point cloud
    │
    ▼  PhNx.Distance
Distance matrix (Nx tensor, Euclidean)
    │
    ▼  PhNx.Filtration
Vietoris-Rips filtration
  (simplices sorted by birth = max pairwise distance among vertices)
    │
    ▼  PhNx.BoundaryMatrix
Sparse boundary matrix over F₂
  (columns = simplices, rows = codimension-1 faces, stored as MapSets)
    │
    ▼  PhNx.Reduction
Column reduction (standard persistence algorithm)
  (pivot elimination via XOR / symmetric difference)
    │
    ▼  PhNx.Persistence
Persistence pairs + essential classes
  → barcode / persistence diagram
```

### Key concepts

- **Vietoris-Rips filtration**: a simplex σ = {v₀, …, vₖ} is born at `max{ dist(vᵢ, vⱼ) }` — the diameter of its vertex set. Simplices are processed in birth order.
- **Boundary matrix**: the m×m matrix ∂ over F₂ where ∂[i,j] = 1 if simplex i is a codimension-1 face of simplex j. Stored sparsely as `%{col → MapSet of rows}`.
- **Column reduction**: reduce ∂ by left-to-right column additions (XOR). A pivot pair (i, j) means simplex i creates a homology class that simplex j destroys. Unpivoted simplices with zero columns are essential (infinite bars).
- **Persistence pair** (i, j): birth = filtration value of simplex i, death = filtration value of simplex j. The homology dimension is `dim(i)`.

## Validation vs ripser

Tested against [ripser](https://github.com/Ripser/ripser) on the SO(3) point cloud (`o3_1024.txt`, 50-point subset, 9-dimensional):

- **49 H₀ pairs**: exact match ✓
- **51 H₁ pairs**: exact match ✓

```
../ripser/ripser --format point-cloud --dim 1 o3_50.txt
```

## Performance

Benchmarked on 50 points × 9 dimensions, max_dim=2 (~20,875 simplices):

| Backend | Distance matrix | Filtration | Boundary | Reduction | Total |
|---|---|---|---|---|---|
| EXLA (CPU) | 1ms | 30ms | 20ms | 554ms | ~620ms |
| BinaryBackend | 19ms | 30ms | 22ms | 554ms | ~640ms |

EXLA accelerates the distance matrix computation ~19× but the bottleneck is the pure-Elixir column reduction (~89% of runtime), which does not use Nx tensors and is unaffected by the backend choice.

## Modules

| Module | Responsibility |
|---|---|
| `PhNx` | Public API (`compute/2`, `print_barcode/1`, `betti_numbers/1`, `most_persistent/2`) |
| `PhNx.Distance` | Nx-powered pairwise Euclidean distance matrix |
| `PhNx.Filtration` | Vietoris-Rips filtration construction |
| `PhNx.BoundaryMatrix` | Sparse boundary matrix over F₂ |
| `PhNx.Reduction` | Standard persistence algorithm (column reduction) |
| `PhNx.Persistence` | Pairs extraction, barcode formatting |

## Limitations & future work

- **Scale**: the naive reduction is O(m³) worst-case in the number of simplices m. For point clouds larger than ~100 points without a threshold, runtime grows quickly.
- **Threshold**: pass `threshold: t` to `compute/2` to limit the filtration and keep runtimes manageable.
- **Parallel reduction**: parallel persistence algorithms (e.g. apparent pairs, cohomology) could bring the reduction step onto Nx tensors and benefit from EXLA/GPU.
- **Sparse distance input**: currently only Euclidean point clouds are supported; sparse or precomputed distance matrices could be added.
