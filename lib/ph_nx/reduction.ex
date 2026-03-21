defmodule PhNx.Reduction do
  @moduledoc """
  Standard persistence algorithm: column reduction of the boundary matrix over F₂.

  Algorithm (Edelsbrunner, Letscher, Zomorodian 2002):
    For each column j (in order):
      While col j has a pivot that is already the pivot of some earlier column i:
        Add column i to column j (XOR, i.e. symmetric difference)
      If col j is nonzero, record pivot(j) → j as a persistence pair.

  A persistence pair (i, j) means:
    - Simplex i creates a homology class (birth = filtration[i].birth)
    - Simplex j destroys that class  (death = filtration[j].birth)

  Simplices that are never the pivot of any column are "essential": they create
  homology classes that persist to infinity.
  """

  alias PhNx.BoundaryMatrix

  @doc """
  Reduce the boundary matrix and return persistence pairs and essential simplices.

  Returns:
    `%{pairs: [{i, j}], essential: [i], reduced: boundary_map}`

  where `i` and `j` are filtration indices.
  """
  def reduce(boundary, filtration_size) do
    # pivot_col: maps a row index (pivot) to the column index that owns it
    {reduced, pivot_col, pairs} =
      Enum.reduce(0..(filtration_size - 1), {boundary, %{}, []}, fn j, {bnd, pivot_col, pairs} ->
        {bnd, pivot_col, pairs} = reduce_column(bnd, pivot_col, pairs, j)
        {bnd, pivot_col, pairs}
      end)

    # Essential simplices: those whose column is zero AND are not a pivot row
    pivot_rows = MapSet.new(Map.keys(pivot_col))
    paired_as_birth = MapSet.new(Enum.map(pairs, fn {i, _j} -> i end))

    essential =
      Enum.filter(0..(filtration_size - 1), fn i ->
        # i is essential if its column reduced to zero AND it was never killed
        not Map.has_key?(reduced, i) and not MapSet.member?(pivot_rows, i) and
          not MapSet.member?(paired_as_birth, i)
      end)

    %{pairs: pairs, essential: essential, reduced: reduced}
  end

  defp reduce_column(boundary, pivot_col, pairs, j) do
    case BoundaryMatrix.lowest(boundary, j) do
      nil ->
        # Column j is already zero — no pair generated here
        {boundary, pivot_col, pairs}

      low ->
        case Map.get(pivot_col, low) do
          nil ->
            # This pivot row is free — record (low, j) as a persistence pair.
            # Clearing lemma: column `low` will reduce to zero when reached,
            # so remove it now to skip that work entirely (O(1) saving per pair).
            boundary = Map.delete(boundary, low)
            {boundary, Map.put(pivot_col, low, j), [{low, j} | pairs]}

          i ->
            # Pivot row `low` is owned by column i — eliminate
            boundary = BoundaryMatrix.add_columns(boundary, j, i)
            reduce_column(boundary, pivot_col, pairs, j)
        end
    end
  end
end
