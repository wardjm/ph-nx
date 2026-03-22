defmodule PhNx.BoundaryMatrix do
  @moduledoc """
  Builds the boundary matrix for a filtration.

  The boundary matrix ∂ is an (m×m) matrix over F₂ where m is the number of simplices.
  Entry ∂[i, j] = 1 if simplex i is a codimension-1 face of simplex j.

  We represent it sparsely as a map `%{col_index => MapSet of row_indices}`.
  Only nonzero columns are stored. Zero columns are absent from the map.
  """

  alias PhNx.Filtration

  @doc """
  Build the sparse boundary matrix from a filtration.

  Returns `{boundary_matrix, filtration}` where `boundary_matrix` is
  `%{non_neg_integer() => MapSet.t(non_neg_integer())}`.
  """
  def build(filtration) do
    index_map = Filtration.index_map(filtration)

    boundary =
      filtration
      |> Enum.reduce(%{}, fn simplex, acc ->
        %{index: j, dim: dim} = simplex

        if dim == 0 do
          # Vertices have empty boundary
          acc
        else
          face_indices =
            simplex
            |> Filtration.faces()
            |> Enum.map(fn face_verts -> Map.fetch!(index_map, face_verts) end)
            |> MapSet.new()

          Map.put(acc, j, face_indices)
        end
      end)

    {boundary, filtration}
  end

  @doc """
  Return the lowest (maximum) row index in column `col`, or `nil` if the column is zero.
  """
  def lowest(boundary, col) do
    case Map.get(boundary, col) do
      nil -> nil
      set -> if MapSet.size(set) == 0, do: nil, else: Enum.max(set)
    end
  end

  @doc """
  Add column `src` to column `dst` over F₂ (symmetric difference of their row sets).
  Returns updated boundary matrix.
  """
  def add_columns(boundary, dst, src) do
    col_dst = Map.get(boundary, dst, MapSet.new())
    col_src = Map.get(boundary, src, MapSet.new())
    result = MapSet.symmetric_difference(col_dst, col_src)

    if MapSet.size(result) == 0 do
      Map.delete(boundary, dst)
    else
      Map.put(boundary, dst, result)
    end
  end

  @doc """
  Convert the sparse boundary matrix to a dense Nx tensor for inspection/visualization.
  Returns an {m, m} tensor over u8.
  """
  def to_tensor(boundary, m) do
    rows = List.duplicate(List.duplicate(0, m), m)

    dense =
      Enum.reduce(boundary, rows, fn {col, row_set}, acc ->
        Enum.reduce(row_set, acc, fn row, a ->
          List.update_at(a, row, fn r -> List.replace_at(r, col, 1) end)
        end)
      end)

    Nx.tensor(dense, type: :u8)
  end
end
