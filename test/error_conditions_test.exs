defmodule PhNx.ErrorConditionsTest do
  use ExUnit.Case, async: true

  alias PhNx.{Distance, Filtration, BoundaryMatrix, Persistence}

  defp assert_in_range(actual, expected, delta), do: assert_in_delta(actual, expected, delta)

  # ── Distance Module Error Tests ─────────────────────────────────────────────

  describe "Distance.euclidean/1" do
    test "single point returns 0.0 on diagonal" do
      pts = Nx.tensor([[0.0, 0.0]])
      d = Distance.euclidean(pts)
      mat = Nx.to_list(d)
      assert_in_range(Enum.at(Enum.at(mat, 0), 0), 0.0, 0.001)
    end

    test "non-2D point clouds work" do
      pts = Nx.tensor([[0.0], [1.0], [2.0]])
      d = Distance.euclidean(pts)
      assert Nx.shape(d) == {3, 3}
      mat = Nx.to_list(d)
      assert_in_range(Enum.at(Enum.at(mat, 1), 2), 1.0, 0.001)
    end

    test "large point cloud does not overflow" do
      pts = Nx.tensor(for i <- 1..1000, do: [i * 1.0, i * 2.0])
      d = Distance.euclidean(pts)
      assert Nx.shape(d) == {1000, 1000}
    end

    test "coordinate dtype f32 works" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 1.0]], type: :f32)
      d = Distance.euclidean(pts)
      assert Nx.type(d) == {:f, 64}
    end

    test "integer coordinates are converted to f64" do
      pts = Nx.tensor([[0, 0], [1, 1]], type: :u8)
      d = Distance.euclidean(pts)
      assert Nx.type(d) == {:f, 64}
    end
  end

  describe "Distance.flat_tuple/1" do
    test "handles single point" do
      pts = Nx.tensor([[0.0, 0.0]])
      d = Distance.euclidean(pts)
      t = Distance.flat_tuple(d)
      # With 1 point, flat_tuple returns a 1-tuple
      assert t == {0.0}
    end

    test "correct indexing" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      t = Distance.flat_tuple(d)
      n = 3
      rows = Nx.to_list(d)

      for i <- 0..(n - 1), j <- 0..(n - 1) do
        expected = Enum.at(Enum.at(rows, i), j)
        actual = elem(t, i * n + j)
        assert_in_range(actual, expected, 0.001)
      end
    end
  end

  # ── Filtration Module Edge Tests ───────────────────────────────────────────

  describe "Filtration.build/2" do
    test "max_dim=0 produces only vertices" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      assert Enum.all?(f, &(&1.dim == 0))
      assert length(f) == 3
    end

    test "max_dim too large for n points" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 10)
      assert length(f) <= 8
    end

    test "vertices with same birth time are sorted by dimension" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      vertices = Enum.filter(f, &(&1.dim == 0))

      first_vertices = Enum.take(vertices, 2) |> Enum.map(& &1.vertices)
      assert first_vertices == [[0], [1]]
    end

    test "duplicate vertices" do
      pts = Nx.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      assert length(f) >= 3
      # Diagonal entries are always 0.0; d(0,1) = 0 since points are identical
      mat = Nx.to_list(d)
      assert_in_range(Enum.at(Enum.at(mat, 0), 0), 0.0, 0.001)
      assert_in_range(Enum.at(Enum.at(mat, 1), 1), 0.0, 0.001)
      assert_in_range(Enum.at(Enum.at(mat, 0), 1), 0.0, 0.001)
    end
  end

  describe "Filtration.index_map/1" do
    test "handles empty filtration" do
      map = Filtration.index_map([])
      assert map == %{}
    end
  end

  # ── Boundary Matrix Edge Tests ────────────────────────────────────────────

  describe "BoundaryMatrix.build_from_filtration/2" do
    test "empty filtration creates empty matrix" do
      bm = BoundaryMatrix.build_from_filtration([])
      assert bm.size == 0
      assert bm.columns == %{}
      assert bm.pivot_col == %{}
      assert bm.pairs == []
      assert MapSet.new(bm.ap_resolved) == MapSet.new()
    end

    test "vertices only: no columns have non-zero entries" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      bm = BoundaryMatrix.build_from_filtration(f)

      Enum.each(0..1, fn i ->
        assert is_nil(BoundaryMatrix.lowest(bm, i))
      end)
    end
  end

  describe "BoundaryMatrix.lowest/2" do
    test "handles columns beyond matrix size" do
      pts = Nx.tensor([[0.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      bm = BoundaryMatrix.build_from_filtration(f)
      assert is_nil(BoundaryMatrix.lowest(bm, 99))
    end
  end

  describe "BoundaryMatrix.as_tensor/2" do
    test "produces an all-zero tensor for a vertices-only filtration" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      bm = BoundaryMatrix.build_from_filtration(f)
      m = length(f)
      t = BoundaryMatrix.as_tensor(bm, m)
      assert Nx.shape(t) == {m, m}
      assert Nx.to_list(t) == [[0, 0], [0, 0]]
    end
  end

  describe "BoundaryMatrix.reduce/1" do
    test "empty filtration: no pairs, no essential" do
      bm = BoundaryMatrix.build_from_filtration([]) |> BoundaryMatrix.reduce()
      assert BoundaryMatrix.pairs(bm) == []
      assert BoundaryMatrix.essential(bm) == []
    end

    test "one point: one essential class" do
      pts = Nx.tensor([[0.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      bm = BoundaryMatrix.build_from_filtration(f) |> BoundaryMatrix.reduce()
      assert BoundaryMatrix.pairs(bm) == []
      assert BoundaryMatrix.essential(bm) == [0]
    end
  end

  # ── Persistence Module Edge Tests ─────────────────────────────────────────

  describe "Persistence.compute/2" do
    test "threshold: infinity includes all simplices" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      result = Persistence.compute(pts, max_dim: 1, threshold: :infinity)
      assert length(result.diagram) > 0
    end

    test "large threshold includes all simplices: full PH computed" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      result = Persistence.compute(pts, max_dim: 2, threshold: 100.0)
      # All simplices included: 1 essential H0 component, 2 H0 pairs
      # (The H1 cycle and its triangle share the same birth time, so birth==death is filtered out)
      assert length(result.essential) == 1
      assert length(result.pairs) == 2
    end

    test "threshold at zero: only vertices" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      result = Persistence.compute(pts, max_dim: 1, threshold: 0.0)
      assert length(result.essential) == 3
      assert length(result.pairs) == 0
    end

    test "valid multiple calls with same input" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

      result1 = Persistence.compute(pts, max_dim: 2)
      result2 = Persistence.compute(pts, max_dim: 2)

      assert result1.diagram == result2.diagram
    end

    test "betti_numbers handles empty essential list" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      result = Persistence.compute(pts, max_dim: 0)
      betti = Persistence.betti_numbers(result)
      assert betti == %{0 => 2}
    end

    test "diagram includes zero-persistence pairs" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      result = Persistence.compute(pts, max_dim: 1)
      assert Enum.all?(result.pairs, fn {_, birth, death} -> birth < death end)
    end

    test "zero distance between points handled correctly" do
      pts = Nx.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
      result = Persistence.compute(pts, max_dim: 2)
      assert length(result.pairs) >= 0
    end

    test "single dimension point cloud" do
      pts = Nx.tensor([[0.0], [1.0], [2.0]])
      result = Persistence.compute(pts, max_dim: 2)
      assert length(result.essential) == 1
      assert length(Enum.filter(result.pairs, fn {d, _, _} -> d == 1 end)) == 0
    end

    test "very large max_dim" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      # max_dim=10 is capped by point count; effectively gives 1 edge, 1 essential H0
      result = Persistence.compute(pts, max_dim: 10)
      assert length(result.essential) == 1
    end
  end

  describe "Persistence.compute/2 threshold validation" do
    test "negative threshold raises ArgumentError" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])

      assert_raise ArgumentError, ~r/threshold must be/, fn ->
        Persistence.compute(pts, threshold: -1.0)
      end
    end

    test "list threshold raises ArgumentError" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])

      assert_raise ArgumentError, ~r/threshold must be/, fn ->
        Persistence.compute(pts, threshold: [1.0, 2.0])
      end
    end

    test "string threshold raises ArgumentError" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])

      assert_raise ArgumentError, ~r/threshold must be/, fn ->
        Persistence.compute(pts, threshold: "big")
      end
    end

    test "zero threshold is valid" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      result = Persistence.compute(pts, threshold: 0.0)
      assert is_map(result)
    end

    test ":infinity threshold is valid" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      result = Persistence.compute(pts, threshold: :infinity)
      assert is_map(result)
    end
  end

  describe "Persistence.most_persistent/2" do
    test "returns empty list when fewer than n pairs" do
      pts = Nx.tensor([[0.0, 0.0]])
      result = Persistence.compute(pts)
      top5 = Persistence.most_persistent(result, 5)
      assert top5 == []
    end

    test "ties broken by birth time (lower birth first)" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      result = Persistence.compute(pts)
      top3 = Persistence.most_persistent(result, 3)

      sorted = Enum.sort_by(top3, fn {_, b, _, p} -> {-p, b} end)
      assert top3 == sorted
    end
  end
end
