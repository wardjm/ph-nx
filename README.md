# ph-nx

Persistent homology in Elixir using [Nx](https://github.com/elixir-nx/nx).

Computes the persistent homology of a finite point cloud via the Vietoris-Rips filtration and the standard persistence algorithm over F₂. Results match [ripser](https://github.com/Ripser/ripser) exactly.

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
result = PhNx.compute(points, max_dim: 2)
# => %{pairs: [{dim, birth, death}], essential: [{dim, birth}], diagram: [...]}

PhNx.print_barcode(result)
PhNx.betti_numbers(result)     # => %{0 => 1, 1 => 0}
PhNx.most_persistent(result, 5) # => [{dim, birth, death, persistence}, ...]
```

### Options for `compute/2`

| Option | Default | Description |
|---|---|---|
| `:max_dim` | `2` | Maximum simplex dimension. To detect Hₖ features you need simplices up to dimension k+1. |
| `:threshold` | enclosing radius | Ignore simplices born after this filtration value. The enclosing radius is the smallest value at which all points are connected. Pass `:infinity` to include all simplices. |

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

- **Scale**: reduction is O(m³) worst-case in the number of simplices. The default threshold (enclosing radius) limits the filtration automatically, but passing `threshold: :infinity` on large point clouds will grow quickly.
- **Sparse distance input**: currently only Euclidean point clouds are supported; sparse or precomputed distance matrices could be added.
