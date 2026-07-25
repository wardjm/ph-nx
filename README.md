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

H2 (1 bar):
  [1.4142, ∞)
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
PhNx.betti_numbers(result)          # => %{0 => 1, 1 => 0}
PhNx.most_persistent(result, 5)     # => [{dim, birth, death, persistence}, ...]
PhNx.filtration_builder(points)     # => [{vertices, birth}, ...]
PhNx.filtration_builder(points, max_dim: 1)
```

### Options for `compute/2`

| Option | Default | Description |
|---|---|---|
| `:max_dim` | `2` | Maximum simplex dimension. To detect Hₖ features you need simplices up to dimension k+1. |
| `:threshold` | enclosing radius | Ignore simplices born after this filtration value. The enclosing radius is the smallest value at which all points are connected. Pass `:infinity` to include all simplices. |
| `:boundary_builder` | `BoundaryMatrix.build_from_filtration/2` | Override the boundary matrix constructor. Useful for testing and alternative implementations. Must be a 2-arity function `(filtration, opts) -> BoundaryMatrix.t()`. |
| `:coeff` | `:z2` | Coefficient ring for the reduction. Pass `{:zp, p}` for ℤₚ arithmetic with signed boundary operators. |
| `:on_progress` | none | 1-arity callback invoked once per column during reduction, receiving `%{current: non_neg_integer(), total: pos_integer()}`. |
| `:backend` | `:cpu` | Distance computation backend (`:cpu` or `:gpu`). `PhNx.compute/2` only — not accepted by `PhNx.Persistence.compute/2`. |

## Modules

| Module | Responsibility |
|---|---|
| `PhNx` | Public API (`compute/2`, `print_barcode/1`, `betti_numbers/1`, `most_persistent/2`, `filtration_builder/1,2`, `reduction/1,2`) |
| `PhNx.Topology` | Façade orchestrating Distance → FiltrationBuilder → Reduction |
| `PhNx.Distance` | Nx-powered pairwise Euclidean distance matrix |
| `PhNx.FiltrationBuilder` | Vietoris-Rips filtration with options |
| `PhNx.Filtration` | Low-level filtration construction |
| `PhNx.BoundaryMatrix` | Sparse boundary matrix over F₂ |
| `PhNx.Reduction` | Standard persistence algorithm (column reduction) |
| `PhNx.Persistence` | Pairs extraction, barcode formatting |

## Command-line tool

A Mix task is provided for exploring the library from the command line.

```
mix ph_nx <file> [--max-dim N] [--threshold T]
```

The input file should contain one point per line with coordinates comma-separated. Blank lines and lines starting with `#` are ignored.

```
# unit square
0.0,0.0
1.0,0.0
1.0,1.0
0.0,1.0
```

```
$ mix ph_nx square.txt

Computing persistent homology for 4 points in 2D...

Persistence Barcode
──────────────────────────────────────────────────

H0 (4 bars):
  [0.0000, 1.0000)  persistence: 1.0000
  [0.0000, 1.0000)  persistence: 1.0000
  [0.0000, 1.0000)  persistence: 1.0000
  [0.0000, ∞)

H1 (1 bar):
  [1.0000, 1.4142)  persistence: 0.4142

H2 (1 bar):
  [1.4142, ∞)

Betti numbers:
  β0 = 1
```

Points can be in any dimension as long as all points in the file have the same number of coordinates. The `--max-dim` and `--threshold` options correspond directly to the options for `PhNx.compute/2`.

Points can also be streamed on stdin with `--stream`:

```
$ cat square.txt | ph_nx --stream
```

### Standalone escript

```
$ mix escript.build
$ ./ph_nx square.txt
```

The escript is a single self-contained file that needs only an Erlang runtime on the target machine.

### Backends used by the CLI

| Entry point | Backend |
|---|---|
| `mix ph_nx` (dev) | `EXLA.Backend`, falling back to `Nx.BinaryBackend` with a warning on stderr if EXLA cannot be loaded |
| `./ph_nx` (escript) | `Nx.BinaryBackend` |

`mix escript.build` builds with `MIX_ENV=prod`, where EXLA is neither configured nor bundled, so the escript computes on `Nx.BinaryBackend`. An escript is a single archive: the `priv/` directory of `:exla` is never unpacked to disk and its NIF (`libexla.so`) cannot be loaded, so accelerated backends and escripts are mutually exclusive. (Forcing a build with `MIX_ENV=dev` still runs — the CLI detects that EXLA is unusable and falls back with a warning — but gains nothing.)

Results are identical either way; only speed differs. Use `mix ph_nx` or the library API with EXLA configured when acceleration matters.

## License

MIT — see [LICENSE](LICENSE) for details.

## Limitations & future work

- **Scale**: reduction is O(m³) worst-case in the number of simplices. The default threshold (enclosing radius) limits the filtration automatically, but passing `threshold: :infinity` on large point clouds will grow quickly. See [docs/performance.md](docs/performance.md) for complexity details, benchmark numbers, and guidance on controlling simplex count.
- **Sparse distance input**: currently only Euclidean point clouds are supported; sparse or precomputed distance matrices could be added.
