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

  Apparent pairs pre-seeded in the `BoundaryMatrix` (via `BoundaryMatrix.from_filtration/2`)
  are used to skip columns that are already resolved, accelerating the reduction.

  ## Reference

  Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2002).
  Topological persistence and simplification.
  *Discrete & Computational Geometry*, 28(4), 511–533.

  ## Example

      points = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      dist = PhNx.Distance.euclidean(points)
      filtration = PhNx.Filtration.build(dist, 2)
      bm = PhNx.BoundaryMatrix.from_filtration(filtration)
      %{pairs: pairs, essential: essential} = PhNx.Reduction.reduce(bm)
      # pairs:    [{1, 3}, {2, 4}, {5, 6}]  — birth/death index pairs
      # essential: [0]                       — vertex 0 generates the lone H0 class
  """

  alias PhNx.BoundaryMatrix

  @typedoc "A raw index pair {birth_index, death_index} from the reduction."
  @type index_pair :: {non_neg_integer(), non_neg_integer()}

  @typedoc "Result of the reduction: raw index pairs, essential simplex indices, and the reduced matrix."
  @type reduction_result :: %{
          pairs: [index_pair()],
          essential: [non_neg_integer()],
          reduced: BoundaryMatrix.t()
        }

  @doc """
  Reduce the boundary matrix and return persistence pairs and essential simplices.

  The `BoundaryMatrix` must be constructed via `BoundaryMatrix.from_filtration/2`.
  Apparent pairs pre-seeded in the matrix are respected — those columns are skipped.

  Returns:
    `%{pairs: [{i, j}], essential: [i], reduced: %BoundaryMatrix{}}`

  where `i` and `j` are filtration indices.
  """
  @spec reduce(BoundaryMatrix.t()) :: reduction_result()
  def reduce(%BoundaryMatrix{} = bm) do
    %BoundaryMatrix{
      columns: columns,
      size: size,
      pivot_col: ap_pivot_col,
      pairs: ap_pairs,
      ap_resolved: ap_resolved,
      ap_deaths: ap_deaths
    } = bm

    {reduced_cols, pivot_col, pairs} =
      Enum.reduce(0..(size - 1), {columns, ap_pivot_col, ap_pairs}, fn j,
                                                                       {cols, pc, ps} ->
        if MapSet.member?(ap_resolved, j) do
          {cols, pc, ps}
        else
          reduce_column(cols, pc, ps, j, ap_deaths)
        end
      end)

    pivot_rows = MapSet.new(Map.keys(pivot_col))
    paired_as_birth = MapSet.new(Enum.map(pairs, fn {i, _j} -> i end))

    essential =
      Enum.filter(0..(size - 1), fn i ->
        not Map.has_key?(reduced_cols, i) and
          not MapSet.member?(pivot_rows, i) and
          not MapSet.member?(paired_as_birth, i) and
          not MapSet.member?(ap_resolved, i)
      end)

    reduced = %{bm | columns: reduced_cols, pairs: pairs, paired_as_birth: paired_as_birth}
    %{pairs: pairs, essential: essential, reduced: reduced}
  end

  defp reduce_column(columns, pivot_col, pairs, j, protected) do
    case col_lowest(columns, j) do
      nil ->
        {columns, pivot_col, pairs}

      low ->
        case Map.get(pivot_col, low) do
          nil ->
            columns =
              if MapSet.member?(protected, low),
                do: columns,
                else: Map.delete(columns, low)

            {columns, Map.put(pivot_col, low, j), [{low, j} | pairs]}

          i ->
            columns = xor_columns(columns, j, i)
            reduce_column(columns, pivot_col, pairs, j, protected)
        end
    end
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
