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

  alias PhNx.{BoundaryMatrix, Filtration}

  @typedoc "A raw index pair {birth_index, death_index} from the reduction."
  @type index_pair :: {non_neg_integer(), non_neg_integer()}

  @typedoc "Result of the reduction: raw index pairs, essential simplex indices, and the reduced matrix."
  @type reduction_result :: %{
          pairs: [index_pair()],
          essential: [non_neg_integer()],
          reduced: BoundaryMatrix.t()
        }

  @doc """
  Pre-pass: identify apparent pairs from the boundary matrix and filtration.

  A pair (σ, τ) is apparent when:
    1. σ = lowest(column(τ))  — σ is the highest-indexed face of τ
    2. τ = min(cofaces of σ)  — τ is the lowest-indexed coface of σ

  Condition 2 uses the MINIMUM coface index. For the forward boundary reduction
  algorithm (columns processed left-to-right), τ = min cofacet of σ guarantees
  no earlier column can have claimed σ as a pivot, so the pair requires zero
  column operations to detect.

  Returns `{pairs, boundary}` where apparent pairs are pre-recorded.
  """
  @spec apparent_pairs(BoundaryMatrix.t(), [Filtration.simplex()]) ::
          {[index_pair()], BoundaryMatrix.t()}
  def apparent_pairs(boundary, filtration) do
    coface_map = build_coface_map(filtration)

    {pairs, boundary} =
      filtration
      |> Enum.reduce({[], boundary}, fn %{index: j}, {pairs, bnd} ->
        with low when not is_nil(low) <- BoundaryMatrix.lowest(bnd, j),
             cofaces when cofaces != [] <- Map.get(coface_map, low, []),
             ^j <- Enum.min(cofaces) do
          # Do NOT delete any columns: a simplex can simultaneously be the death
          # of one ap pair (H0) and the birth of another (H1). Its column must
          # remain in the boundary so other columns can eliminate against it.
          # The skip set (ap_resolved) prevents re-processing; the ap_deaths
          # protection set prevents the clearing lemma from deleting it.
          {[{low, j} | pairs], bnd}
        else
          _ -> {pairs, bnd}
        end
      end)

    {pairs, boundary}
  end

  defp build_coface_map(filtration) do
    vertex_to_idx = Filtration.index_map(filtration)

    Enum.reduce(filtration, %{}, fn simplex, acc ->
      simplex
      |> Filtration.faces()
      |> Enum.reduce(acc, fn face_verts, a ->
        case Map.get(vertex_to_idx, face_verts) do
          nil ->
            a

          face_idx ->
            Map.update(a, face_idx, [simplex.index], &[simplex.index | &1])
        end
      end)
    end)
  end

  @doc """
  Reduce the boundary matrix and return persistence pairs and essential simplices.

  Returns:
    `%{pairs: [{i, j}], essential: [i], reduced: boundary_map}`

  where `i` and `j` are filtration indices.
  """
  @spec reduce(BoundaryMatrix.t(), non_neg_integer(), [Filtration.simplex()] | nil) ::
          reduction_result()
  def reduce(boundary, filtration_size, filtration \\ nil) do
    {ap_pairs, boundary} =
      if filtration, do: apparent_pairs(boundary, filtration), else: {[], boundary}

    # Seed pivot_col with apparent pairs so the main loop can eliminate against them.
    # Track births+deaths to skip, and death columns to protect from clearing.
    {ap_pivot_col, ap_resolved, ap_deaths} =
      Enum.reduce(ap_pairs, {%{}, MapSet.new(), MapSet.new()}, fn {low, j}, {pc, res, deaths} ->
        {Map.put(pc, low, j), res |> MapSet.put(low) |> MapSet.put(j), MapSet.put(deaths, j)}
      end)

    # pivot_col: maps a row index (pivot) to the column index that owns it
    {reduced, pivot_col, pairs} =
      Enum.reduce(0..(filtration_size - 1), {boundary, ap_pivot_col, ap_pairs}, fn j,
                                                                                   {bnd,
                                                                                    pivot_col,
                                                                                    pairs} ->
        if MapSet.member?(ap_resolved, j) do
          {bnd, pivot_col, pairs}
        else
          {bnd, pivot_col, pairs} = reduce_column(bnd, pivot_col, pairs, j, ap_deaths)
          {bnd, pivot_col, pairs}
        end
      end)

    # Essential simplices: those whose column is zero AND are not a pivot row
    # ap_resolved covers both births and deaths of apparent pairs.
    pivot_rows = MapSet.new(Map.keys(pivot_col))
    paired_as_birth = MapSet.new(Enum.map(pairs, fn {i, _j} -> i end))

    essential =
      Enum.filter(0..(filtration_size - 1), fn i ->
        not Map.has_key?(reduced, i) and
          not MapSet.member?(pivot_rows, i) and
          not MapSet.member?(paired_as_birth, i) and
          not MapSet.member?(ap_resolved, i)
      end)

    %{pairs: pairs, essential: essential, reduced: reduced}
  end

  defp reduce_column(boundary, pivot_col, pairs, j, protected) do
    case BoundaryMatrix.lowest(boundary, j) do
      nil ->
        {boundary, pivot_col, pairs}

      low ->
        case Map.get(pivot_col, low) do
          nil ->
            # Pivot row is free — record pair.
            # Clearing lemma: delete birth column unless it's a protected ap-death.
            boundary =
              if MapSet.member?(protected, low),
                do: boundary,
                else: Map.delete(boundary, low)

            {boundary, Map.put(pivot_col, low, j), [{low, j} | pairs]}

          i ->
            boundary = BoundaryMatrix.add_columns(boundary, j, i)
            reduce_column(boundary, pivot_col, pairs, j, protected)
        end
    end
  end
end
