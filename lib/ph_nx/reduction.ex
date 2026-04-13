defmodule PhNx.Reduction do
  @moduledoc """
  The core implementation of the sparse F₂ boundary matrix reduction algorithm.
  This module owns the heavy lifting of the column reduction process,
  responsible for computing persistence pairs from a filtration.
  """

  alias PhNx.BoundaryMatrix

  @doc """
  Reduces a boundary matrix or a filtration to find persistence pairs.

  If a `BoundaryMatrix` is provided, it performs the standard column-reduction algorithm.
  If a filtration (list of simplices) is provided, it first builds the boundary matrix
  from the filtration and then performs the reduction.

  Returns the reduced boundary matrix.
  """
  @spec reduce(BoundaryMatrix.t() | list(PhNx.Filtration.simplex())) :: BoundaryMatrix.t()
  def reduce(%BoundaryMatrix{} = bm) do
    BoundaryMatrix.reduce(bm)
  end

  def reduce(filtration) when is_list(filtration) do
    filtration
    |> BoundaryMatrix.build_from_filtration()
    |> BoundaryMatrix.reduce()
  end

  def reduce(other) do
    raise ArgumentError, "PhNx.Reduction.reduce/1 expects a BoundaryMatrix or a list of simplices, but got #{inspect(other)}"
  end
end
