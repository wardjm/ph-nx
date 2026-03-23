defmodule PhNx.BoundaryMatrix do
  @moduledoc """
  Builds the boundary matrix for a filtration.

  The boundary matrix ∂ is an (m×m) matrix over F₂ where m is the number of simplices.
  Entry ∂[i, j] = 1 if simplex i is a codimension-1 face of simplex j.

  We represent it sparsely as a map `%{col_index => MapSet of row_indices}`.
  Only nonzero columns are stored. Zero columns are absent from the map.

  ## Example

      points = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      dist = PhNx.Distance.euclidean(points)
      filtration = PhNx.Filtration.build(dist, 1)
      {boundary, _} = PhNx.BoundaryMatrix.build(filtration)
      # Vertices have no boundary (empty columns omitted).
      # Each edge column holds the indices of its two endpoint vertices:
      # %{3 => MapSet([0, 1]), 4 => MapSet([0, 2]), 5 => MapSet([1, 2])}
  """

  alias PhNx.Filtration

  @typedoc "Sparse boundary matrix over F₂: maps column index to the set of nonzero row indices."
  @type t :: %{optional(non_neg_integer()) => MapSet.t(non_neg_integer())}

  @doc """
  Build the sparse boundary matrix from a filtration.

  Returns `{boundary_matrix, filtration}` where `boundary_matrix` is
  `%{non_neg_integer() => MapSet.t(non_neg_integer())}`.
  """
  @spec build([Filtration.simplex()]) :: {t(), [Filtration.simplex()]}
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
  @spec lowest(t(), non_neg_integer()) :: non_neg_integer() | nil
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
  @spec add_columns(t(), non_neg_integer(), non_neg_integer()) :: t()
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
  @spec to_tensor(t(), non_neg_integer()) :: Nx.Tensor.t()
  def to_tensor(boundary, m) do
    base = Nx.broadcast(Nx.tensor(0, type: :u8), {m, m})

    all_indices =
      Enum.flat_map(boundary, fn {col, row_set} ->
        Enum.map(row_set, fn row -> [row, col] end)
      end)

    if all_indices == [] do
      base
    else
      indices = Nx.tensor(all_indices)
      updates = Nx.broadcast(Nx.tensor(1, type: :u8), {length(all_indices)})
      Nx.indexed_put(base, indices, updates)
    end
  end
end
