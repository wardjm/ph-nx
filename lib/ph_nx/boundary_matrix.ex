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
      # Edges can be inspected via as_tensor/2 or lowest/2.
  """

  alias PhNx.Filtration

  # All struct fields below are implementation details and are NOT part of the public API.
  # Use pairs/1, essential/1, and as_tensor/2 to access results.
  @opaque t :: %__MODULE__{
            columns: %{optional(non_neg_integer()) => MapSet.t(non_neg_integer())},
            size: non_neg_integer(),
            pivot_col: %{optional(non_neg_integer()) => non_neg_integer()},
            pairs: [{non_neg_integer(), non_neg_integer()}],
            ap_resolved: MapSet.t(non_neg_integer()),
            ap_deaths: MapSet.t(non_neg_integer()),
            reduced: boolean()
          }

  @type column_result :: :already_resolved | :zero | :paired

  defstruct columns: %{},
            size: 0,
            pivot_col: %{},
            pairs: [],
            ap_resolved: MapSet.new(),
            ap_deaths: MapSet.new(),
            reduced: false

  @doc """
  Build a BoundaryMatrix from a filtration.

  By default, runs the apparent-pairs pre-pass to seed `pivot_col`, `pairs`,
  `ap_resolved`, and `ap_deaths`, so that `reduce/1` can skip those
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
  @spec as_tensor(t(), non_neg_integer()) :: Nx.Tensor.t()
  def as_tensor(%__MODULE__{columns: cols}, m) do
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

  @doc """
  Reduce all columns and return the finished matrix.

  This is the full column-reduction loop over all columns. Apparent pairs
  pre-seeded by `from_filtration/2` are respected — those columns are skipped.

  Also accepts a filtration list as a convenience; equivalent to
  `from_filtration(filtration) |> reduce()`. The list overload always uses
  default `from_filtration/2` options (apparent pairs seeded).
  """
  @spec reduce(t()) :: t()
  @spec reduce([Filtration.simplex()]) :: t()
  def reduce(%__MODULE__{size: size} = bm) do
    result =
      Enum.reduce(0..(size - 1)//1, bm, fn col, acc ->
        {_result, acc} = reduce_column(acc, col)
        acc
      end)

    %{result | reduced: true}
  end

  def reduce(filtration) when is_list(filtration) do
    filtration |> from_filtration() |> reduce()
  end

  @doc """
  Return the persistence pairs from a fully reduced boundary matrix.

  Raises `ArgumentError` if called on an unreduced matrix.
  """
  @spec pairs(t()) :: [{non_neg_integer(), non_neg_integer()}]
  def pairs(%__MODULE__{reduced: false}),
    do:
      raise(ArgumentError, "pairs/1 called on an unreduced BoundaryMatrix — call reduce/1 first")

  def pairs(%__MODULE__{pairs: p}), do: p

  @doc """
  Return the essential simplex indices from a fully reduced boundary matrix.

  Essential simplices are those that create homology classes persisting to infinity.
  Raises `ArgumentError` if called on an unreduced matrix.
  """
  @spec essential(t()) :: [non_neg_integer()]
  def essential(%__MODULE__{reduced: false}),
    do:
      raise(
        ArgumentError,
        "essential/1 called on an unreduced BoundaryMatrix — call reduce/1 first"
      )

  def essential(%__MODULE__{size: size} = bm) do
    pivot_rows = MapSet.new(Map.keys(bm.pivot_col))

    Enum.filter(0..(size - 1)//1, fn i ->
      not Map.has_key?(bm.columns, i) and
        not MapSet.member?(pivot_rows, i) and
        not MapSet.member?(bm.ap_resolved, i)
    end)
  end

  @doc """
  Extract persistence pairs and essential simplex indices from a fully reduced matrix.

  **Must be called after `reduce/1`.** Calling this on an unreduced matrix returns
  silently incorrect results: only apparent pairs will appear in `pairs`, and
  `essential` will be heavily overcounted.

  Returns `{pairs, essential}` where:
    * `pairs` is `[{birth_index, death_index}]` in reverse reduction order — sort if
      stable output is required
    * `essential` is `[simplex_index]` — simplices that create classes persisting to infinity

  The four-condition essential-class filter is applied here and nowhere else.
  """
  @spec result(t()) :: {[{non_neg_integer(), non_neg_integer()}], [non_neg_integer()]}
  def result(%__MODULE__{size: size} = bm) do
    # Invariant: keys(pivot_col) = apparent-pair births ∪ reduction-found births.
    # Apparent pairs seed pivot_col in from_filtration/2; reduction pairs add to pivot_col
    # in do_reduce_column/2. Both paths land in pivot_col, so a single membership check
    # here covers all paired births.
    # ap_resolved additionally excludes apparent-pair death columns, which can remain in
    # bm.columns when the death index is also a birth in another pair.
    pivot_rows = MapSet.new(Map.keys(bm.pivot_col))

    essential =
      Enum.filter(0..(size - 1)//1, fn i ->
        not Map.has_key?(bm.columns, i) and
          not MapSet.member?(pivot_rows, i) and
          not MapSet.member?(bm.ap_resolved, i)
      end)

    {bm.pairs, essential}
  end

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
                pairs: [{low, col} | bm.pairs]
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
