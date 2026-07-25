defmodule PhNx.BoundaryMatrix do
  @moduledoc """
  Opaque boundary matrix for a filtration.

  ## Public interface

  Build a boundary matrix with `build_from_filtration/2`, reduce it with `reduce/1`,
  then read results via `pairs/1` and `essential/1`.

      points = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      dist = PhNx.Distance.euclidean(points)
      filtration = PhNx.Filtration.build(dist, 1)
      bm = filtration |> PhNx.BoundaryMatrix.build_from_filtration() |> PhNx.BoundaryMatrix.reduce()
      pairs = PhNx.BoundaryMatrix.pairs(bm)
      essential = PhNx.BoundaryMatrix.essential(bm)

  ## Implementation details (opaque)

  The reduction algorithm, internal struct shape, and all intermediate state
  (`pivot_col`, `ap_resolved`, `ap_deaths`) are implementation details and are
  not part of the public API. The matrix is stored sparsely as a map of column
  index to the set of nonzero row indices; an apparent-pairs pre-pass seeds
  the pivot map before the standard column-reduction loop runs.

  For coefficient rings other than ℤ₂ (specified via `coeff: {:zp, p}`), columns
  are stored as maps from row index to coefficient, and reduction uses modular
  scalar arithmetic instead of XOR.
  """

  alias PhNx.Filtration

  # All struct fields below are implementation details and are NOT part of the public API.
  # Use pairs/1, essential/1, and as_tensor/2 to access results.
  @opaque t :: %__MODULE__{
            columns: %{optional(non_neg_integer()) => MapSet.t(non_neg_integer())},
            zp_columns: %{
              optional(non_neg_integer()) => %{optional(non_neg_integer()) => pos_integer()}
            },
            coeff_ring: :z2 | {:zp, pos_integer()},
            size: non_neg_integer(),
            pivot_col: %{optional(non_neg_integer()) => non_neg_integer()},
            pairs: [{non_neg_integer(), non_neg_integer()}],
            ap_resolved: MapSet.t(non_neg_integer()),
            ap_deaths: MapSet.t(non_neg_integer()),
            reduced: boolean()
          }

  @type column_result :: :already_resolved | :zero | :paired

  defstruct columns: %{},
            zp_columns: %{},
            coeff_ring: :z2,
            size: 0,
            pivot_col: %{},
            pairs: [],
            ap_resolved: MapSet.new(),
            ap_deaths: MapSet.new(),
            reduced: false

  @doc """
  Build a BoundaryMatrix from a filtration.

  This is the canonical public entry point for constructing a boundary matrix.

  By default, runs the apparent-pairs pre-pass to seed `pivot_col`, `pairs`,
  `ap_resolved`, and `ap_deaths`, so that `reduce/1` can skip those
  columns entirely.

  Options:
    * `seed_apparent: false` — skip the apparent-pairs pre-pass. Produces a
      bare matrix useful for testing reduction without the pre-pass optimisation.
    * `coeff: :z2` — use ℤ₂ coefficients (default).
    * `coeff: {:zp, p}` — use ℤₚ coefficients with signed boundary operators.
      The apparent-pairs pre-pass is skipped for non-ℤ₂ rings.
  """
  @spec build_from_filtration([Filtration.simplex()], keyword()) :: t()
  def build_from_filtration(filtration, opts \\ []) do
    case Keyword.get(opts, :coeff, :z2) do
      :z2 -> from_filtration(filtration, Keyword.delete(opts, :coeff))
      {:zp, p} -> from_filtration_zp(filtration, p)
    end
  end

  defp from_filtration(filtration, opts) do
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

  defp from_filtration_zp(filtration, p) do
    index_map = Filtration.index_map(filtration)
    zp_cols = build_zp_columns(filtration, index_map, p)
    %__MODULE__{zp_columns: zp_cols, size: length(filtration), coeff_ring: {:zp, p}}
  end

  @doc """
  Return the lowest (maximum) row index in column `col`, or `nil` if the column is zero.
  """
  @spec lowest(t(), non_neg_integer()) :: non_neg_integer() | nil
  def lowest(%__MODULE__{coeff_ring: :z2, columns: cols}, col), do: col_lowest(cols, col)

  def lowest(%__MODULE__{coeff_ring: {:zp, _}, zp_columns: zp_cols}, col) do
    case Map.get(zp_cols, col) do
      nil -> nil
      m when map_size(m) == 0 -> nil
      m -> Enum.max(Map.keys(m))
    end
  end

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
  Returns `true` if `term` is a boundary matrix.

  `t()` is opaque, so callers outside this module cannot pattern-match the
  struct to dispatch on it — doing so breaks opacity and Dialyzer reports it.
  This predicate is the supported way to ask.
  """
  @spec boundary_matrix?(term()) :: boolean()
  def boundary_matrix?(%__MODULE__{}), do: true
  def boundary_matrix?(_term), do: false

  @doc """
  Reduce all columns and return the finished matrix.

  This is the full column-reduction loop over all columns. Apparent pairs
  pre-seeded by `build_from_filtration/2` are respected — those columns are skipped.

  Also accepts a filtration list as a convenience; equivalent to
  `build_from_filtration(filtration) |> reduce()`. The list overload always uses
  default `build_from_filtration/2` options (apparent pairs seeded).
  """
  @spec reduce(t()) :: t()
  @spec reduce(t(), keyword()) :: t()
  @spec reduce([Filtration.simplex()]) :: t()
  def reduce(bm_or_filtration, opts \\ [])

  def reduce(%__MODULE__{coeff_ring: :z2, size: size} = bm, opts) do
    on_progress = Keyword.get(opts, :on_progress)

    result =
      Enum.reduce(0..(size - 1)//1, bm, fn col, acc ->
        if on_progress, do: on_progress.(%{current: col, total: size})
        {_result, acc} = reduce_column(acc, col)
        acc
      end)

    %{result | reduced: true}
  end

  def reduce(%__MODULE__{coeff_ring: {:zp, p}, size: size} = bm, opts) do
    on_progress = Keyword.get(opts, :on_progress)

    result =
      Enum.reduce(0..(size - 1)//1, bm, fn col, acc ->
        if on_progress, do: on_progress.(%{current: col, total: size})
        {_result, acc} = do_reduce_zp_column(acc, col, p)
        acc
      end)

    %{result | reduced: true}
  end

  def reduce(filtration, opts) when is_list(filtration) do
    filtration |> build_from_filtration() |> reduce(opts)
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

  def essential(%__MODULE__{} = bm), do: do_essential(bm)

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

  defp do_reduce_zp_column(%__MODULE__{zp_columns: zp_cols} = bm, col, p) do
    case zp_lowest(zp_cols, col) do
      nil ->
        {:zero, bm}

      {low, c_current} ->
        case Map.get(bm.pivot_col, low) do
          nil ->
            new_bm = %{
              bm
              | pivot_col: Map.put(bm.pivot_col, low, col),
                pairs: [{low, col} | bm.pairs]
            }

            {:paired, new_bm}

          pivot_col_j ->
            c_pivot = get_in(zp_cols, [pivot_col_j, low])
            # scalar r: c_current + r * c_pivot ≡ 0 (mod p)
            r = Integer.mod(p - Integer.mod(c_current * mod_inverse(c_pivot, p), p), p)

            new_col =
              add_cols_zp(Map.get(zp_cols, col, %{}), Map.get(zp_cols, pivot_col_j, %{}), r, p)

            new_zp_cols =
              if map_size(new_col) == 0,
                do: Map.delete(zp_cols, col),
                else: Map.put(zp_cols, col, new_col)

            do_reduce_zp_column(%{bm | zp_columns: new_zp_cols}, col, p)
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

  defp build_zp_columns(filtration, index_map, p) do
    Enum.reduce(filtration, %{}, fn
      %{dim: 0}, acc -> acc
      %{index: j, vertices: verts}, acc -> Map.put(acc, j, zp_column(verts, index_map, p))
    end)
  end

  # The boundary of a k-simplex is the alternating sum of its faces: dropping
  # vertex k contributes coefficient (-1)^k, which in ℤₚ is 1 or p-1.
  defp zp_column(verts, index_map, p) do
    verts
    |> Enum.with_index()
    |> Enum.into(%{}, fn {_, k} ->
      face_idx = Map.fetch!(index_map, List.delete_at(verts, k))
      {face_idx, if(rem(k, 2) == 0, do: 1, else: p - 1)}
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

  defp zp_lowest(zp_cols, col) do
    case Map.get(zp_cols, col) do
      nil ->
        nil

      m when map_size(m) == 0 ->
        nil

      m ->
        row = Enum.max(Map.keys(m))
        {row, Map.fetch!(m, row)}
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

  defp add_cols_zp(col_a, col_b, scalar, p) do
    all_rows = Enum.uniq(Map.keys(col_a) ++ Map.keys(col_b))

    Enum.reduce(all_rows, %{}, fn row, acc ->
      val = Integer.mod(Map.get(col_a, row, 0) + scalar * Map.get(col_b, row, 0), p)
      if val == 0, do: acc, else: Map.put(acc, row, val)
    end)
  end

  defp mod_inverse(a, p) do
    # Fermat's little theorem: a^(p-2) ≡ a⁻¹ (mod p) for prime p
    pow_mod(a, p - 2, p)
  end

  defp pow_mod(_base, 0, _mod), do: 1

  defp pow_mod(base, exp, mod) do
    if rem(exp, 2) == 0 do
      half = pow_mod(base, div(exp, 2), mod)
      Integer.mod(half * half, mod)
    else
      Integer.mod(base * pow_mod(base, exp - 1, mod), mod)
    end
  end

  # Invariant: keys(pivot_col) = apparent-pair births ∪ reduction-found births.
  # Apparent pairs seed pivot_col in build_from_filtration/2; reduction pairs add to pivot_col
  # in do_reduce_column/2. Both paths land in pivot_col, so a single membership check
  # here covers all paired births.
  # ap_resolved additionally excludes apparent-pair death columns, which can remain in
  # bm.columns when the death index is also a birth in another pair.
  defp do_essential(%__MODULE__{coeff_ring: :z2, size: size} = bm) do
    pivot_rows = MapSet.new(Map.keys(bm.pivot_col))

    Enum.filter(0..(size - 1)//1, fn i ->
      not Map.has_key?(bm.columns, i) and
        not MapSet.member?(pivot_rows, i) and
        not MapSet.member?(bm.ap_resolved, i)
    end)
  end

  defp do_essential(%__MODULE__{coeff_ring: {:zp, _}, size: size} = bm) do
    pivot_rows = MapSet.new(Map.keys(bm.pivot_col))

    Enum.filter(0..(size - 1)//1, fn i ->
      not Map.has_key?(bm.zp_columns, i) and not MapSet.member?(pivot_rows, i)
    end)
  end
end
