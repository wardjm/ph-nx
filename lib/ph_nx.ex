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
    - `PhNx.BoundaryMatrix` — sparse boundary matrix over F₂, including column reduction
    - `PhNx.Persistence`    — high-level API and output formatting
  """

  @spec compute(Nx.Tensor.t() | [[number()]], keyword()) :: PhNx.Persistence.result()
  defdelegate compute(points, opts \\ []), to: PhNx.Persistence

  @spec print_barcode(PhNx.Persistence.result()) :: :ok
  defdelegate print_barcode(result), to: PhNx.Persistence

  @spec most_persistent(PhNx.Persistence.result(), pos_integer()) ::
          [{non_neg_integer(), float(), float(), float()}]
  defdelegate most_persistent(result, n \\ 10), to: PhNx.Persistence

  @spec betti_numbers(PhNx.Persistence.result()) ::
          %{optional(non_neg_integer()) => non_neg_integer()}
  defdelegate betti_numbers(result), to: PhNx.Persistence
end
