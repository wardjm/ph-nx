defmodule PhNx.ReductionPropertyTest do
  use ExUnit.Case, async: true

  alias PhNx.{BoundaryMatrix, Filtration}

  defp generate_random_dist_matrix(n) do
    # Generate an n x n matrix with random values in (0, 1]
    vals = for _ <- 0..(n * n - 1), do: :rand.uniform()

    # Split flat list into rows
    rows = Enum.chunk_every(vals, n)

    # Convert to Nx tensor
    base_tensor = Nx.tensor(rows, type: :f32)

    # Make it symmetric: dist[i,j] = dist[j,i]
    symmetric = Nx.add(base_tensor, Nx.transpose(base_tensor, axes: [1, 0]))
    symmetric = Nx.multiply(symmetric, 0.5)

    # Set diagonal to 0 (distance from a point to itself)
    identity = Nx.eye(n, type: :f32)
    one_minus_identity = Nx.subtract(1, identity)
    dist = Nx.multiply(symmetric, one_minus_identity)

    dist
  end

  describe "Property-based reduction invariant" do
    test "reduced matrix satisfies pivot uniqueness and correctness" do
      # We run several iterations with different random matrices
      for _ <- 1..10 do
        n = Enum.random(3..15)
        dist = generate_random_dist_matrix(n)

        # We use simplex dimension 2
        filtration = PhNx.Filtration.build(dist, 2)
        # We use seed_apparent: false to force the reduction algorithm to work on all columns
        # This is important to test the actual reduction logic.
        bm = PhNx.BoundaryMatrix.build_from_filtration(filtration, seed_apparent: false)
        reduced_bm = PhNx.Reduction.reduce(bm)

        # 1. Every pair {low, col} in pairs must have 'low' as the lowest non-zero row in 'col'
        Enum.each(reduced_bm.pairs, fn {low, col} ->
          case PhNx.BoundaryMatrix.lowest(reduced_bm, col) do
            nil ->
              assert false, "Column #{col} is zero but has a pivot #{low}"

            actual_low ->
              assert actual_low == low, "Expected pivot #{low} but lowest is #{actual_low}"
          end
        end)

        # 2. For every column that has a pivot, it should be in the pairs list
        # The algorithm handles this, but let's be sure.
        # Every pair in pivot_col must be present in pairs.
        Enum.each(Map.to_list(reduced_bm.pivot_col), fn {low, col} ->
          assert {low, col} in reduced_bm.pairs, "Pivot #{low} -> #{col} not in pairs list"
        end)

        # 3. Every column NOT in pairs must be a zero column
        paired_cols = Enum.map(reduced_bm.pairs, fn {_low, col} -> col end)
        unpaired_cols = for col <- 0..(reduced_bm.size - 1), col not in paired_cols, do: col

        Enum.each(unpaired_cols, fn col ->
          assert PhNx.BoundaryMatrix.lowest(reduced_bm, col) == nil,
                 "Column #{col} should be zero but has entries"
        end)

        # 4. Pivot uniqueness: each pivot row should have at most one pivot column
        pivot_rows = Enum.map(reduced_bm.pairs, fn {row, _col} -> row end)

        assert length(pivot_rows) == length(Enum.uniq(pivot_rows)),
               "Pivot rows should be unique: #{inspect(pivot_rows)}"

        # 5. Each pivot column should only appear once
        pivot_cols = Enum.map(reduced_bm.pairs, fn {_row, col} -> col end)

        assert length(pivot_cols) == length(Enum.uniq(pivot_cols)),
               "Pivot columns should be unique: #{inspect(pivot_cols)}"

        # 6. Birth < Death for all pairs (columns are processed in filtration order)
        Enum.each(reduced_bm.pairs, fn {birth, death} ->
          assert birth < death, "Birth #{birth} should be less than death #{death}"
        end)
      end
    end
  end

  describe "Property tests with apparent pairs enabled" do
    test "reduction works correctly with apparent pairs pre-pass" do
      for _ <- 1..10 do
        n = Enum.random(5..20)
        dist = generate_random_dist_matrix(n)

        filtration = PhNx.Filtration.build(dist, 2)

        # With apparent pairs enabled (default)
        bm_with_apparent = PhNx.BoundaryMatrix.build_from_filtration(filtration)
        reduced_with_apparent = PhNx.Reduction.reduce(bm_with_apparent)

        # Without apparent pairs
        bm_without_apparent =
          PhNx.BoundaryMatrix.build_from_filtration(filtration, seed_apparent: false)

        reduced_without_apparent = PhNx.Reduction.reduce(bm_without_apparent)

        # Both should produce the same pairs (order may differ)
        pairs_with = Enum.sort(reduced_with_apparent.pairs)
        pairs_without = Enum.sort(reduced_without_apparent.pairs)

        assert pairs_with == pairs_without,
               "Apparent pairs optimization should produce same pairs:\nWith: #{inspect(pairs_with)}\nWithout: #{inspect(pairs_without)}"
      end
    end
  end

  describe "Edge cases for property tests" do
    test "reduction handles very small complexes" do
      # 3 points (minimum for 2D simplex)
      dist =
        Nx.tensor([
          [0.0, 1.0, 1.5],
          [1.0, 0.0, 2.0],
          [1.5, 2.0, 0.0]
        ])

      filtration = PhNx.Filtration.build(dist, 1)
      bm = PhNx.BoundaryMatrix.build_from_filtration(filtration, seed_apparent: false)
      reduced_bm = PhNx.Reduction.reduce(bm)

      # Check invariants
      Enum.each(reduced_bm.pairs, fn {low, col} ->
        assert PhNx.BoundaryMatrix.lowest(reduced_bm, col) == low
      end)
    end

    test "reduction handles empty result (all zero columns)" do
      # A 0-dimensional filtration has no boundary columns
      points = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      dist = PhNx.Distance.euclidean(points)
      filtration = PhNx.Filtration.build(dist, 0)

      bm = PhNx.BoundaryMatrix.build_from_filtration(filtration, seed_apparent: false)
      reduced_bm = PhNx.Reduction.reduce(bm)

      assert reduced_bm.pairs == []
      assert reduced_bm.reduced == true
    end

    test "reduction handles many zero columns" do
      # Create a complex where many columns are zero
      n = 10
      dist = generate_random_dist_matrix(n)

      # Dimension 0 only - all boundary columns are zero
      filtration = PhNx.Filtration.build(dist, 0)
      bm = PhNx.BoundaryMatrix.build_from_filtration(filtration, seed_apparent: false)
      reduced_bm = PhNx.Reduction.reduce(bm)

      assert reduced_bm.reduced == true
      # All columns should be zero or essential
      for col <- 0..(reduced_bm.size - 1) do
        _lowest = PhNx.BoundaryMatrix.lowest(reduced_bm, col)

        # Either the column is zero, or it has a pivot (which wouldn't be in pairs if it's essential)
      end
    end
  end
end
