defmodule PhNx.BoundaryMatrix do
  @moduledoc """
  Opaque boundary matrix for a filtration, with pre-seeded apparent pairs.

  The boundary matrix ∂ is an (m×m) matrix over F₂ where m is the number of simplices.
  Entry ∂[i, j] = 1 if simplex i is a codimension-1 face of simplex j.

  Internally the matrix is stored sparsely as a map of column index to the set of nonzero
  row indices. All reduction state (`pivot_col`, `pairs`, `ap_resolved`, `ap_deaths`,
  `paired_as_birth`) is co-located in the same struct but is not accessible to callers.

  ## Example

      points = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      dist = PhNx.Distance.euclidean(points)
      filtration = PhNx.Filtration.build(dist, 1)
      bm = PhNx.BoundaryMatrix.from_filtration(filtration)
      # Vertices have no boundary; apparent pairs are pre-seeded automatically.
      # Edges can be inspected via to_tensor/2 or lowest/2.
  """

  alias PhNx.Filtration

  @opaque t :: %__MODULE__{
            columns: %{optional(non_neg_integer()) => MapSet.t(non_neg_integer())},
            size: non_neg_integer(),
            pivot_col: %{optional(non_neg_integer()) => non_neg_integer()},
            pairs: [{non_neg_integer(), non_neg_integer()}],
            ap_resolved: MapSet.t(non_neg_integer()),
            ap_deaths: MapSet.t(non_neg_integer()),
            paired_as_birth: MapSet.t(non_neg_integer())
          }

  defstruct columns: %{},
            size: 0,
            pivot_col: %{},
            pairs: [],
            ap_resolved: MapSet.new(),
            ap_deaths: MapSet.new(),
            paired_as_birth: MapSet.new()

  @doc """
  Build a BoundaryMatrix from a filtration.

  By default, runs the apparent-pairs pre-pass to seed `pivot_col`, `pairs`,
  `ap_resolved`, and `ap_deaths`, so that `Reduction.reduce/1` can skip those
  columns entirely.

  Options:
    * `seed_apparent: false` — skip the apparent-pairs pre-pass. Produces a
      bare matrix useful for testing reduction without the pre-pass optimisation.
  """
  @spec from_filtration([Filtration.simplex()], keyword()) :: t()
  def from_filtration(filtration, opts \\ []) do
    seed_apparent = Keyword.get(opts, :seed_apparent, true)
    index_map = Filtration.index_map(filtration)
    columns = build_columns(filtration, index_map)
    size = length(filtration)
    base = %__MODULE__{columns: columns, size: size}

    if seed_apparent do
      coface_map = build_coface_map(filtration, index_map)
      pairs = find_apparent_pairs(columns, filtration, coface_map)

      {pivot_col, ap_resolved, ap_deaths} =
        Enum.reduce(pairs, {%{}, MapSet.new(), MapSet.new()}, fn {low, j}, {pc, res, deaths} ->
          {Map.put(pc, low, j), res |> MapSet.put(low) |> MapSet.put(j), MapSet.put(deaths, j)}
        end)

      %{base | pivot_col: pivot_col, pairs: pairs, ap_resolved: ap_resolved, ap_deaths: ap_deaths}
    else
      base
    end
  end

  @doc """
  Return the lowest (maximum) row index in column `col`, or `nil` if the column is zero.
  """
  @spec lowest(t(), non_neg_integer()) :: non_neg_integer() | nil
  def lowest(%__MODULE__{columns: cols}, col), do: col_lowest(cols, col)

  @doc """
  Convert the sparse boundary matrix to a dense Nx tensor for inspection/visualization.
  Returns an {m, m} tensor over u8.
  """
  @spec to_tensor(t(), non_neg_integer()) :: Nx.Tensor.t()
  def to_tensor(%__MODULE__{columns: cols}, m) do
    base = Nx.broadcast(Nx.tensor(0, type: :u8), {m, m})

    all_indices =
      Enum.flat_map(cols, fn {col, row_set} ->
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

  @type column_result :: :already_resolved | :zero | :paired

  @doc """
  Perform one step of the standard column reduction algorithm on column `col`.

  Returns `{result, updated_matrix}` where `result` is one of:
    * `:already_resolved` — the column was pre-resolved by an apparent pair
    * `:zero` — the column is (or reduces to) zero; no pair recorded
    * `:paired` — the column reduced to a nonzero pivot; a pair was recorded
  """
  @spec reduce_column(t(), non_neg_integer()) :: {column_result(), t()}
  def reduce_column(%__MODULE__{ap_resolved: ap_resolved} = bm, col) do
    if MapSet.member?(ap_resolved, col) do
      {:already_resolved, bm}
    else
      do_reduce_column(bm, col)
    end
  end

  defp do_reduce_column(%__MODULE__{columns: cols} = bm, col) do
    case col_lowest(cols, col) do
      nil ->
        {:zero, bm}

      low ->
        case Map.get(bm.pivot_col, low) do
          nil ->
            # No existing pivot — record the pair
            protected = MapSet.member?(bm.ap_deaths, low)

            new_cols =
              if protected,
                do: bm.columns,
                else: Map.delete(bm.columns, low)

            new_bm = %{
              bm
              | columns: new_cols,
                pivot_col: Map.put(bm.pivot_col, low, col),
                pairs: [{low, col} | bm.pairs],
                paired_as_birth: MapSet.put(bm.paired_as_birth, low)
            }

            {:paired, new_bm}

          pivot_col ->
            # XOR with the existing pivot column and retry
            new_cols = xor_columns(bm.columns, col, pivot_col)
            do_reduce_column(%{bm | columns: new_cols}, col)
        end
    end
  end

  # ── Private helpers ────────────────────────────────────────────────────────

  defp build_columns(filtration, index_map) do
    Enum.reduce(filtration, %{}, fn simplex, acc ->
      %{index: j, dim: dim} = simplex

      if dim == 0 do
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
  end

  defp build_coface_map(filtration, index_map) do
    Enum.reduce(filtration, %{}, fn simplex, acc ->
      simplex
      |> Filtration.faces()
      |> Enum.reduce(acc, fn face_verts, a ->
        case Map.get(index_map, face_verts) do
          nil -> a
          face_idx -> Map.update(a, face_idx, [simplex.index], &[simplex.index | &1])
        end
      end)
    end)
  end

  defp find_apparent_pairs(columns, filtration, coface_map) do
    Enum.reduce(filtration, [], fn %{index: j}, pairs ->
      with low when not is_nil(low) <- col_lowest(columns, j),
           cofaces when cofaces != [] <- Map.get(coface_map, low, []),
           ^j <- Enum.min(cofaces) do
        [{low, j} | pairs]
      else
        _ -> pairs
      end
    end)
  end

  defp col_lowest(columns, col) do
    case Map.get(columns, col) do
      nil -> nil
      set -> if MapSet.size(set) == 0, do: nil, else: Enum.max(set)
    end
  end

  defp xor_columns(columns, dst, src) do
    col_dst = Map.get(columns, dst, MapSet.new())
    col_src = Map.get(columns, src, MapSet.new())
    result = MapSet.symmetric_difference(col_dst, col_src)

    if MapSet.size(result) == 0,
      do: Map.delete(columns, dst),
      else: Map.put(columns, dst, result)
  end
end
