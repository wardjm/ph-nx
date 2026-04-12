defmodule PhNx.Reduction do
  @moduledoc """
  The core implementation of the sparse F₂ boundary matrix reduction algorithm.
  This module owns the heavy lifting of the column reduction process,
  responsible for computing persistence pairs from a filtration.
  """

  alias PhNx.BoundaryMatrix

  @doc """
  Reduces a boundary matrix to find persistence pairs.

  This function takes a boundary matrix (which may or may or not have been pre-seeded
  with apparent pairs) and performs the standard column-reduction algorithm.

  Returns the reduced boundary matrix.
  """
  @spec reduce(BoundaryMatrix.t()) :: BoundaryMatrix.t()
  def reduce(%BoundaryMatrix{} = bm) do
    BoundaryMatrix.reduce(bm)
  end

  @doc """
  Reduces a filtration directly, returning the reduced boundary matrix.
  """
  @spec reduce(list(PhNx.Filtration.simplex())) :: BoundaryMatrix.t()
  def reduce(filtration) do
    filtration
    |> BoundaryMatrix.build_from_filtration()
    |> BoundaryMatrix.reduce()
  end
end
