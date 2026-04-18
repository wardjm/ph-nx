defmodule PhNx.Reduction do
  @moduledoc """
  Boundary matrix reduction for persistent homology.

  Supports ℤ₂ (default) and arbitrary ℤₚ coefficient rings. Pass `coeff: {:zp, p}` to
  `reduce/2` to use ℤₚ arithmetic with signed boundary operators.
  """

  alias PhNx.BoundaryMatrix

  @doc """
  Reduces a boundary matrix or a filtration to find persistence pairs.

  If a `BoundaryMatrix` is provided, it performs the standard column-reduction algorithm.
  If a filtration (list of simplices) is provided, it first builds the boundary matrix
  from the filtration and then performs the reduction.

  Returns the reduced boundary matrix.
  """
  @spec reduce(BoundaryMatrix.t() | list(PhNx.Filtration.simplex()), keyword()) ::
          BoundaryMatrix.t()
  def reduce(input, opts \\ [])

  def reduce(%BoundaryMatrix{} = bm, _opts) do
    BoundaryMatrix.reduce(bm)
  end

  def reduce(filtration, opts) when is_list(filtration) do
    filtration
    |> BoundaryMatrix.build_from_filtration(opts)
    |> BoundaryMatrix.reduce()
  end

  def reduce(other, _opts) do
    raise ArgumentError,
          "PhNx.Reduction.reduce/2 expects a BoundaryMatrix or a list of simplices, but got #{inspect(other)}"
  end
end
