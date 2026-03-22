defmodule PhNxTest do
  use ExUnit.Case, async: true

  alias PhNx.{Distance, Filtration, BoundaryMatrix, Reduction, Persistence}

  # ── Distance ────────────────────────────────────────────────────────────────

  describe "Distance.euclidean/1" do
    test "3-point right triangle" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      assert Nx.shape(d) == {3, 3}

      mat = Nx.to_list(d)
      assert Enum.at(Enum.at(mat, 0), 0) == 0.0
      assert_in_delta Enum.at(Enum.at(mat, 0), 1), 1.0, 1.0e-9
      assert_in_delta Enum.at(Enum.at(mat, 0), 2), 1.0, 1.0e-9
      assert_in_delta Enum.at(Enum.at(mat, 1), 2), :math.sqrt(2), 1.0e-9
    end

    test "symmetry" do
      pts = Nx.tensor([[0.0, 0.0], [3.0, 4.0]])
      d = Distance.euclidean(pts)
      mat = Nx.to_list(d)
      assert_in_delta Enum.at(Enum.at(mat, 0), 1), 5.0, 1.0e-9
      assert_in_delta Enum.at(Enum.at(mat, 1), 0), 5.0, 1.0e-9
    end
  end

  describe "Distance.enclosing_radius/1" do
    test "two points: radius equals their distance" do
      pts = Nx.tensor([[0.0, 0.0], [5.0, 0.0]])
      d = Distance.euclidean(pts)
      assert_in_delta Distance.enclosing_radius(d), 5.0, 1.0e-9
    end

    test "unit square: radius equals diagonal (sqrt 2)" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      assert_in_delta Distance.enclosing_radius(d), :math.sqrt(2), 1.0e-9
    end
  end

  describe "Distance.sorted_edges/1" do
    test "returns edges sorted by length" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      edges = Distance.sorted_edges(d)
      dists = Enum.map(edges, fn {_, _, x} -> x end)
      assert dists == Enum.sort(dists)
    end

    test "returns n*(n-1)/2 edges for n points" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
      d = Distance.euclidean(pts)
      edges = Distance.sorted_edges(d)
      assert length(edges) == 6
    end

    test "each edge {i, j, dist} has i < j and correct distance" do
      pts = Nx.tensor([[0.0, 0.0], [3.0, 4.0]])
      d = Distance.euclidean(pts)
      [{i, j, dist}] = Distance.sorted_edges(d)
      assert i == 0
      assert j == 1
      assert_in_delta dist, 5.0, 1.0e-9
    end
  end

  # ── Filtration ───────────────────────────────────────────────────────────────

  describe "Filtration.build/2" do
    test "3 points, max_dim=1 produces 3 vertices + 3 edges" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      assert length(f) == 6
      assert Enum.count(f, &(&1.dim == 0)) == 3
      assert Enum.count(f, &(&1.dim == 1)) == 3
    end

    test "simplices are sorted: birth non-decreasing" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      births = Enum.map(f, & &1.birth)
      assert births == Enum.sort(births)
    end

    test "indices are 0-based and contiguous" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      indices = Enum.map(f, & &1.index)
      assert indices == Enum.to_list(0..(length(f) - 1))
    end

    test "single point: one vertex, birth 0, nothing else" do
      pts = Nx.tensor([[0.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      assert length(f) == 1
      assert hd(f) == %{vertices: [0], dim: 0, birth: 0.0, index: 0}
    end

    test "vertex birth is 0" do
      pts = Nx.tensor([[0.0, 0.0], [5.0, 5.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      vertices = Enum.filter(f, &(&1.dim == 0))
      assert Enum.all?(vertices, &(&1.birth == 0.0))
    end

    test "edge birth is the distance between its endpoints" do
      # Points: (0,0), (3,4), (0,1)
      # Distances: d(0,1)=5.0, d(0,2)=1.0, d(1,2)=sqrt((3-0)²+(4-1)²)=sqrt(18)
      pts = Nx.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      # idx maps vertex-list key to birth value (Map key access via [])
      idx = Map.new(f, fn s -> {s.vertices, s.birth} end)
      assert_in_delta idx[[0, 1]], 5.0, 1.0e-9
      assert_in_delta idx[[0, 2]], 1.0, 1.0e-9
      assert_in_delta idx[[1, 2]], :math.sqrt(9 + 9), 1.0e-9
    end

    test "triangle birth is the max edge length (diameter of vertex set)" do
      # Right triangle: legs 1.0 and 1.0, hypotenuse sqrt(2)
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      tri = Enum.find(f, &(&1.vertices == [0, 1, 2]))
      assert_in_delta tri.birth, :math.sqrt(2), 1.0e-9
    end
  end

  describe "Filtration.faces/1" do
    test "faces of a triangle are 3 edges" do
      simplex = %{vertices: [0, 1, 2], dim: 2, birth: 1.0, index: 5}
      faces = Filtration.faces(simplex)
      assert length(faces) == 3
      assert [1, 2] in faces
      assert [0, 2] in faces
      assert [0, 1] in faces
    end

    test "faces of an edge are 2 vertices" do
      simplex = %{vertices: [2, 4], dim: 1, birth: 0.5, index: 3}
      faces = Filtration.faces(simplex)
      assert [[4], [2]] == faces
    end
  end

  # ── BoundaryMatrix ───────────────────────────────────────────────────────────

  describe "BoundaryMatrix.build/1" do
    test "vertices have no boundary" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      {bnd, _} = BoundaryMatrix.build(f)
      assert map_size(bnd) == 0
    end

    test "edge boundary has exactly 2 rows set" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      {bnd, _} = BoundaryMatrix.build(f)

      edge_simplices = Enum.filter(f, &(&1.dim == 1))

      Enum.each(edge_simplices, fn %{index: j} ->
        col = Map.get(bnd, j, MapSet.new())
        assert MapSet.size(col) == 2
      end)
    end
  end

  describe "BoundaryMatrix.lowest/2" do
    test "returns max row index" do
      bnd = %{3 => MapSet.new([1, 5, 2])}
      assert BoundaryMatrix.lowest(bnd, 3) == 5
    end

    test "returns nil for missing column" do
      assert BoundaryMatrix.lowest(%{}, 0) == nil
    end

    test "returns nil for empty column set" do
      assert BoundaryMatrix.lowest(%{0 => MapSet.new()}, 0) == nil
    end
  end

  describe "BoundaryMatrix.add_columns/3" do
    test "XOR of identical columns gives empty column (removed from map)" do
      bnd = %{0 => MapSet.new([1, 3]), 1 => MapSet.new([1, 3])}
      result = BoundaryMatrix.add_columns(bnd, 0, 1)
      refute Map.has_key?(result, 0)
    end

    test "XOR produces symmetric difference" do
      bnd = %{0 => MapSet.new([1, 2, 3]), 1 => MapSet.new([2, 3, 4])}
      result = BoundaryMatrix.add_columns(bnd, 0, 1)
      assert MapSet.equal?(Map.fetch!(result, 0), MapSet.new([1, 4]))
    end
  end

  describe "BoundaryMatrix.to_tensor/2" do
    test "empty boundary produces zero matrix" do
      t = BoundaryMatrix.to_tensor(%{}, 3)
      assert Nx.shape(t) == {3, 3}
      assert Nx.to_list(t) == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    end

    test "single entry sets correct cell" do
      bnd = %{2 => MapSet.new([0, 1])}
      t = BoundaryMatrix.to_tensor(bnd, 3)
      mat = Nx.to_list(t)
      # row 0, col 2 and row 1, col 2 should be 1
      assert Enum.at(Enum.at(mat, 0), 2) == 1
      assert Enum.at(Enum.at(mat, 1), 2) == 1
      # all other cells zero
      assert Enum.at(Enum.at(mat, 2), 2) == 0
    end

    test "matches boundary built from a real filtration" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      {bnd, _} = BoundaryMatrix.build(f)
      m = length(f)
      t = BoundaryMatrix.to_tensor(bnd, m)
      assert Nx.shape(t) == {m, m}
      # Each edge column should have exactly 2 ones
      Enum.each(Enum.filter(f, &(&1.dim == 1)), fn %{index: j} ->
        col = t[[.., j]] |> Nx.to_list()
        assert Enum.sum(col) == 2
      end)
    end
  end

  # ── Apparent pairs ───────────────────────────────────────────────────────────

  describe "Reduction.apparent_pairs/2" do
    test "detects apparent pair in two-point filtration" do
      # [0](idx 0), [1](idx 1), [0,1](idx 2)
      # lowest([0,1]) = 1; highest coface of [1] = 2 → apparent pair (1, 2)
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      {bnd, _} = BoundaryMatrix.build(f)
      {pairs, _bnd2} = Reduction.apparent_pairs(bnd, f)
      assert {1, 2} in pairs
    end

    test "returns empty when no apparent pairs exist" do
      # Single vertex — no edges, no pairs possible
      pts = Nx.tensor([[0.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      {bnd, _} = BoundaryMatrix.build(f)
      {pairs, _} = Reduction.apparent_pairs(bnd, f)
      assert pairs == []
    end

    test "does not modify the boundary (columns needed for elimination)" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      {bnd, _} = BoundaryMatrix.build(f)
      {[{_birth, death}], bnd2} = Reduction.apparent_pairs(bnd, f)
      # Death column must be kept so other columns can eliminate against it
      assert Map.has_key?(bnd2, death)
      assert bnd == bnd2
    end
  end

  # ── Reduction ────────────────────────────────────────────────────────────────

  describe "Reduction.reduce/3 (with apparent pairs)" do
    test "produces same pairs and essential as reduce/2" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      {bnd, _} = BoundaryMatrix.build(f)
      r2 = Reduction.reduce(bnd, length(f))
      r3 = Reduction.reduce(bnd, length(f), f)
      assert Enum.sort(r3.pairs) == Enum.sort(r2.pairs)
      assert Enum.sort(r3.essential) == Enum.sort(r2.essential)
    end

    test "apparent pair is recorded directly in result (two-point filtration)" do
      # Filtration: v0(0), v1(1), e01(2). Apparent pair: (1, 2).
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      {bnd, _} = BoundaryMatrix.build(f)
      result = Reduction.reduce(bnd, length(f), f)
      assert {1, 2} in result.pairs
      assert 0 in result.essential
    end
  end

  describe "Reduction.reduce/2" do
    test "single vertex has no pairs, one essential class" do
      pts = Nx.tensor([[0.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      {bnd, _} = BoundaryMatrix.build(f)
      result = Reduction.reduce(bnd, length(f))
      assert result.pairs == []
      assert result.essential == [0]
    end

    test "clearing: every birth column is absent from the reduced map" do
      # For any pair (i, j), column i is pre-cleared after the pair is claimed.
      # This holds for H0 pairs (vertex birth, empty column) and for H1 pairs
      # (edge birth, non-empty column) — the edge column must be gone.
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      {bnd, _} = BoundaryMatrix.build(f)
      %{pairs: pairs, reduced: reduced} = Reduction.reduce(bnd, length(f))

      Enum.each(pairs, fn {i, _j} ->
        refute Map.has_key?(reduced, i),
               "birth column #{i} should be cleared after pairing"
      end)
    end

    test "two isolated vertices: one edge pair kills one H0 class" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      {bnd, _} = BoundaryMatrix.build(f)
      result = Reduction.reduce(bnd, length(f))
      assert length(result.pairs) == 1
      assert length(result.essential) == 1
    end
  end

  # ── Persistence (integration) ────────────────────────────────────────────────

  describe "Persistence.compute/2" do
    test "single point: 1 essential H0, no other features" do
      pts = Nx.tensor([[0.0, 0.0]])
      result = Persistence.compute(pts)
      assert result.essential == [{0, 0.0}]
      assert result.pairs == []
    end

    test "two separate points: 1 essential H0, 1 finite H0" do
      pts = Nx.tensor([[0.0, 0.0], [10.0, 0.0]])
      result = Persistence.compute(pts)
      h0_essential = Enum.filter(result.essential, fn {d, _} -> d == 0 end)
      h0_pairs = Enum.filter(result.pairs, fn {d, _, _} -> d == 0 end)
      assert length(h0_essential) == 1
      assert length(h0_pairs) == 1
    end

    test "unit square: 1 essential H0, at least 1 finite H1 loop" do
      pts =
        Nx.tensor([
          [0.0, 0.0],
          [1.0, 0.0],
          [1.0, 1.0],
          [0.0, 1.0]
        ])

      result = Persistence.compute(pts, max_dim: 2)

      h0_essential = Enum.filter(result.essential, fn {d, _} -> d == 0 end)
      h1_finite = Enum.filter(result.pairs, fn {d, _, _} -> d == 1 end)

      assert length(h0_essential) == 1
      assert length(h1_finite) >= 1
    end

    test "equilateral triangle: H1 loop has zero persistence (born and killed simultaneously)" do
      # All edges and the triangle appear at the same ε, so birth == death
      s = :math.sqrt(3) / 2

      pts =
        Nx.tensor([
          [0.0, 0.0],
          [1.0, 0.0],
          [0.5, s]
        ])

      result = Persistence.compute(pts, max_dim: 2)
      # No finite H1 pair with birth < death
      h1_finite = Enum.filter(result.pairs, fn {d, _, _} -> d == 1 end)
      assert h1_finite == []
      # 1 connected component survives
      assert length(Enum.filter(result.essential, fn {d, _} -> d == 0 end)) == 1
    end

    test "unit square: H1 loop born at 1.0, killed at sqrt(2)" do
      # The 4 boundary edges (length 1) form a loop; diagonals (length sqrt(2))
      # enable triangles that fill the loop.
      pts =
        Nx.tensor([
          [0.0, 0.0],
          [1.0, 0.0],
          [1.0, 1.0],
          [0.0, 1.0]
        ])

      result = Persistence.compute(pts, max_dim: 2)
      h1_finite = Enum.filter(result.pairs, fn {d, _, _} -> d == 1 end)

      assert length(h1_finite) == 1
      [{1, birth, death}] = h1_finite
      assert_in_delta birth, 1.0, 1.0e-9
      assert_in_delta death, :math.sqrt(2), 1.0e-9
    end

    test "betti_numbers returns correct map" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      result = Persistence.compute(pts)
      betti = Persistence.betti_numbers(result)
      assert Map.get(betti, 0, 0) == 1
    end

    test "diagram is union of finite pairs and essential classes" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      result = Persistence.compute(pts)
      expected_count = length(result.pairs) + length(result.essential)
      assert length(result.diagram) == expected_count
    end

    test "empty list raises ArgumentError" do
      assert_raise ArgumentError, "point cloud must be non-empty", fn ->
        Persistence.compute([])
      end
    end

    test "single-point list does not raise" do
      assert %{pairs: _, essential: _, diagram: _} = Persistence.compute([[0.0, 0.0]])
    end

    test "single-point tensor does not raise" do
      pts = Nx.tensor([[0.0, 0.0]])
      assert %{pairs: _, essential: _, diagram: _} = Persistence.compute(pts)
    end

    test "negative max_dim raises ArgumentError" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      assert_raise ArgumentError, ~r/max_dim must be a non-negative integer/, fn ->
        Persistence.compute(pts, max_dim: -1)
      end
    end

    test "float max_dim raises ArgumentError" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      assert_raise ArgumentError, ~r/max_dim must be a non-negative integer/, fn ->
        Persistence.compute(pts, max_dim: 1.5)
      end
    end

    test "max_dim of 0 is valid and returns only H0" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      result = Persistence.compute(pts, max_dim: 0)
      assert Enum.all?(result.essential, fn {d, _} -> d == 0 end)
      assert Enum.all?(result.pairs, fn {d, _, _} -> d == 0 end)
    end

    test "unknown option key raises ArgumentError" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      assert_raise ArgumentError, fn -> Persistence.compute(pts, max_dims: 2) end
    end

    test "multiple unknown option keys raise ArgumentError" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      assert_raise ArgumentError, fn -> Persistence.compute(pts, foo: 1, bar: 2) end
    end

    test "valid options do not raise" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      assert %{pairs: _, essential: _, diagram: _} = Persistence.compute(pts, max_dim: 1)
      assert %{pairs: _, essential: _, diagram: _} = Persistence.compute(pts, threshold: 2.0)
      assert %{pairs: _, essential: _, diagram: _} = Persistence.compute(pts, max_dim: 1, threshold: 2.0)
    end
  end

  describe "Persistence.most_persistent/2" do
    test "returns pairs sorted by persistence descending" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      result = Persistence.compute(pts)
      top = Persistence.most_persistent(result, 5)
      persistences = Enum.map(top, fn {_, _, _, p} -> p end)
      assert persistences == Enum.sort(persistences, :desc)
    end
  end
end
