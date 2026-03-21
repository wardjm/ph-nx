defmodule PhNx do
  @moduledoc """
  Persistent homology using Nx.

  Computes the persistent homology of a finite point cloud via the
  Vietoris-Rips filtration and the standard persistence algorithm over F₂.

  ## Quick start

      # Four corners of a unit square — expect H0: 1 component, H1: 1 loop
      points = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
      result = PhNx.compute(points)
      PhNx.print_barcode(result)

  ## Pipeline

      Distance matrix  →  Vietoris-Rips filtration  →  Boundary matrix  →  Column reduction  →  Persistence pairs

  Modules:
    - `PhNx.Distance`       — Nx-powered pairwise Euclidean distance matrix
    - `PhNx.Filtration`     — Vietoris-Rips filtration construction
    - `PhNx.BoundaryMatrix` — sparse boundary matrix over F₂
    - `PhNx.Reduction`      — standard persistence algorithm
    - `PhNx.Persistence`    — high-level API and output formatting
  """

  defdelegate compute(points, opts \\ []), to: PhNx.Persistence
  defdelegate print_barcode(result), to: PhNx.Persistence
  defdelegate most_persistent(result, n \\ 10), to: PhNx.Persistence
  defdelegate betti_numbers(result), to: PhNx.Persistence
end
