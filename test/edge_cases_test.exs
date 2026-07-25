defmodule PhNx.EdgeCasesTest do
  use ExUnit.Case, async: true

  alias PhNx.{BoundaryMatrix, Distance, Filtration, Persistence}

  defp assert_in_range(actual, expected, delta), do: assert_in_delta(actual, expected, delta)

  # ── Numerical Precision Tests ─────────────────────────────────────────────

  describe "Numerical Precision" do
    test "floating point comparison" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      result = Persistence.compute(pts, max_dim: 2)

      Enum.each(result.diagram, fn {dim, birth, death} ->
        assert is_float(birth)
        assert is_float(death) or death == :infinity
        assert dim >= 0
      end)
    end

    test "very small distances" do
      pts = Nx.tensor([[0.0, 0.0], [1.0e-10, 0.0], [2.0e-10, 0.0]])
      result = Persistence.compute(pts, max_dim: 1)
      # 3 collinear points: default threshold connects them into 1 component
      assert length(result.essential) == 1
    end

    test "very large distances" do
      pts = Nx.tensor([[0.0, 0.0], [1.0e10, 0.0], [2.0e10, 0.0]])
      result = Persistence.compute(pts, max_dim: 1)
      # 3 collinear points: default threshold connects them into 1 component
      assert length(result.essential) == 1
    end

    test "floating point accumulation" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
      result = Persistence.compute(pts, max_dim: 2)
      assert length(result.essential) == 1
    end
  end

  # ── Degenerate Geometries Tests ───────────────────────────────────────────

  describe "Degenerate Geometries" do
    test "all points identical" do
      pts = Nx.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
      result = Persistence.compute(pts, max_dim: 2)
      assert length(result.essential) == 1
    end

    test "collinear points (1D structure)" do
      pts = Nx.tensor([[0.0], [1.0], [2.0], [3.0]])
      result = Persistence.compute(pts, max_dim: 2)
      assert length(result.essential) == 1
      assert Enum.filter(result.pairs, fn {d, _, _} -> d == 1 end) == []
    end

    test "two points: exactly one edge pair" do
      pts = Nx.tensor([[0.0, 0.0], [5.0, 5.0]])
      result = Persistence.compute(pts, max_dim: 1)
      h0_pairs = Enum.filter(result.pairs, fn {d, _, _} -> d == 0 end)
      assert length(h0_pairs) == 1
    end

    test "three collinear points" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
      result = Persistence.compute(pts, max_dim: 1)
      assert length(result.essential) == 1
      assert Enum.filter(result.pairs, fn {d, _, _} -> d == 1 end) == []
    end

    test "three collinear points with max_dim=2" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
      result = Persistence.compute(pts, max_dim: 2)
      assert length(result.essential) == 1
      assert Enum.filter(result.pairs, fn {d, _, _} -> d == 1 end) == []
    end
  end

  # ── Boundary Conditions Tests ─────────────────────────────────────────────

  describe "Boundary Conditions" do
    test "points exactly on unit circle" do
      pts =
        Nx.tensor([
          [1.0, 0.0],
          [0.0, 1.0],
          [-1.0, 0.0],
          [0.0, -1.0]
        ])

      result = Persistence.compute(pts, max_dim: 2)
      # 4 points form a triangulated 2-sphere (boundary of tetrahedron):
      # H0 = 1 essential, H1 = 1 finite pair (loop born sqrt(2), dies at 2.0),
      # H2 = 1 essential (sphere). Total essential = 2.
      assert length(result.essential) == 2
      h1 = Enum.filter(result.pairs, fn {d, _, _} -> d == 1 end)
      assert length(h1) == 1
    end

    test "dense collinear points produce 1 connected component" do
      # 3 tightly-spaced collinear points; enclosing_radius only includes adjacent edges
      pts = Nx.tensor([[0.001, 0.0], [0.002, 0.0], [0.003, 0.0]])
      result = Persistence.compute(pts, max_dim: 2)
      assert length(result.essential) == 1
    end

    test "collinear points in high-dimensional space produce 1 connected component" do
      # 3 collinear 3D points; same topology as the 2D collinear case
      pts = Nx.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
      result = Persistence.compute(pts, max_dim: 2)
      assert length(result.essential) == 1
    end
  end

  # ── Filtration Edge Cases ─────────────────────────────────────────────────

  describe "Filtration.build/2 edge cases" do
    test "birth time computed as max edge for triangle" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      tri = Enum.find(f, &(&1.vertices == [0, 1, 2]))
      assert_in_range(tri.birth, :math.sqrt(2), 0.001)
    end

    test "edge birth is the distance between its endpoints" do
      pts = Nx.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      idx = Map.new(f, fn s -> {s.vertices, s.birth} end)
      assert_in_range(idx[[0, 1]], 5.0, 0.001)
      assert_in_range(idx[[0, 2]], 1.0, 0.001)
      assert_in_range(idx[[1, 2]], :math.sqrt(9 + 9), 0.001)
    end

    test "triangle birth is max edge length" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      tri = Enum.find(f, &(&1.vertices == [0, 1, 2]))
      assert_in_range(tri.birth, :math.sqrt(2), 0.001)
    end

    test "vertices birth is 0" do
      pts = Nx.tensor([[0.0, 0.0], [5.0, 5.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      vertices = Enum.filter(f, &(&1.dim == 0))
      assert Enum.all?(vertices, &(&1.birth == 0.0))
    end
  end

  describe "BoundaryMatrix reduce edge cases" do
    test "nil column returns empty pairs and essential" do
      bm = BoundaryMatrix.build_from_filtration([]) |> BoundaryMatrix.reduce()
      assert BoundaryMatrix.pairs(bm) == []
      assert BoundaryMatrix.essential(bm) == []
    end

    test "essential classes respect four-condition" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)

      bm =
        BoundaryMatrix.build_from_filtration(f, seed_apparent: false) |> BoundaryMatrix.reduce()

      pairs = BoundaryMatrix.pairs(bm)
      essential = BoundaryMatrix.essential(bm)

      Enum.each(essential, fn idx ->
        not_in_pairs = Enum.all?(pairs, fn {i, _j} -> i != idx end)
        assert not_in_pairs, "Vertex #{idx} should not be in any pair"
      end)
    end

    test "pivot columns are not counted as essential" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.build_from_filtration(f)
      reduced = bm |> BoundaryMatrix.reduce()
      essential = BoundaryMatrix.essential(reduced)

      # Vertex 0 is the surviving component (essential); vertex 1 is paired with the edge
      assert 0 in essential
      # Edge column 2 is resolved as an apparent pair — confirmed not in essential
      assert MapSet.member?(bm.ap_resolved, 2)
    end
  end

  describe "Persistence operations on degenerate inputs" do
    test "betti_numbers on identical points" do
      pts = Nx.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
      result = Persistence.compute(pts, max_dim: 2)
      betti = Persistence.betti_numbers(result)
      assert Map.get(betti, 0, 0) == 1
    end

    test "diagram is union of finite pairs and essential classes" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      result = Persistence.compute(pts)
      expected_count = length(result.pairs) + length(result.essential)
      assert length(result.diagram) == expected_count
    end

    test "betti_numbers returns correct map" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      result = Persistence.compute(pts)
      betti = Persistence.betti_numbers(result)
      assert Map.get(betti, 0, 0) == 1
    end
  end

  describe "Filtration.faces/1 edge cases" do
    test "triangle faces are the vertex-deletion subsequences" do
      # faces/1 removes each vertex at its position in turn — no sorting
      simplex = %{vertices: [2, 0, 1], dim: 2, birth: 1.0, index: 5}
      faces = Filtration.faces(simplex)
      assert faces == [[0, 1], [2, 1], [2, 0]]
    end

    test "edge faces" do
      simplex = %{vertices: [1, 3], dim: 1, birth: 0.5, index: 2}
      faces = Filtration.faces(simplex)
      assert faces == [[3], [1]]
    end

    test "1D simplex has exactly 2 faces" do
      simplex = %{vertices: [0, 5], dim: 1, birth: 2.5, index: 10}
      faces = Filtration.faces(simplex)
      assert length(faces) == 2
    end
  end

  describe "Filtration.index_map/1 edge cases" do
    test "correctly maps vertex lists to indices" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      map = Filtration.index_map(f)

      assert map[[0]] == 0
      assert map[[1]] == 1
      assert map[[0, 1]] == 3
    end

    test "missing vertex list returns nil" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      map = Filtration.index_map(f)

      assert map[[0]] == 0
      assert map[[1]] == 1
      assert map[[0, 1]] == 2
      assert is_nil(map[[2]])
      assert is_nil(map[[0, 1, 2]])
    end
  end

  describe "Distance.enclosing_radius/1 edge cases" do
    test "scalar tensor returns radius" do
      d = Nx.tensor([[1.0, 2.0], [2.0, 3.0]])
      r = Distance.enclosing_radius(d)
      assert_in_range(r, 2.0, 0.001)
    end

    test "negative distances? (shouldn't happen in Euclidean)" do
      d = Nx.tensor([[-1.0, 0.0], [0.0, 1.0]])
      r = Distance.enclosing_radius(d)
      assert is_number(r)
      assert r >= 0.0
    end
  end

  # ── BoundaryMatrix.reduce_column/2 ───────────────────────────────────────

  describe "BoundaryMatrix.reduce_column/2" do
    test "already_resolved column returns :already_resolved without mutating the matrix" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.build_from_filtration(f)

      # The edge column (index 2) is resolved as an apparent pair during seeding
      {:already_resolved, bm2} = BoundaryMatrix.reduce_column(bm, 2)
      assert bm2 == bm
    end

    test "vertex column returns :zero" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      bm = BoundaryMatrix.build_from_filtration(f)

      {:zero, bm2} = BoundaryMatrix.reduce_column(bm, 0)
      assert bm2 == bm
    end

    test "edge column without seeding returns :paired and records the pair" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.build_from_filtration(f, seed_apparent: false)

      {:paired, bm2} = BoundaryMatrix.reduce_column(bm, 2)
      assert Map.has_key?(bm2.pivot_col, 1)
      assert {1, 2} in bm2.pairs
    end
  end

  # ── BoundaryMatrix.as_tensor/2 ────────────────────────────────────────────

  describe "BoundaryMatrix.as_tensor/2 shape and content" do
    test "shape is {m, m} for a filtration of size m" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.build_from_filtration(f)
      m = length(f)
      t = BoundaryMatrix.as_tensor(bm, m)
      assert Nx.shape(t) == {m, m}
    end

    test "vertex columns are all-zero" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.build_from_filtration(f, seed_apparent: false)
      m = length(f)
      t = BoundaryMatrix.as_tensor(bm, m)

      Enum.each(0..1, fn i ->
        col = t[[.., i]] |> Nx.to_list()
        assert Enum.sum(col) == 0
      end)
    end

    test "edge column has exactly 2 ones" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.build_from_filtration(f, seed_apparent: false)
      t = BoundaryMatrix.as_tensor(bm, 3)

      col = t[[.., 2]] |> Nx.to_list()
      assert Enum.count(col, &(&1 == 1)) == 2
    end
  end

  # ── BoundaryMatrix.reduce/1 correctness ───────────────────────────────────

  describe "BoundaryMatrix.reduce/1 correctness" do
    test "full triangle has 3 pairs (2 H0 + 1 H1) and 1 essential class" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      bm = BoundaryMatrix.build_from_filtration(f) |> BoundaryMatrix.reduce()
      assert length(BoundaryMatrix.pairs(bm)) == 3
      assert length(BoundaryMatrix.essential(bm)) == 1
    end
  end

  # ── Persistence.print_barcode/1 ───────────────────────────────────────────

  describe "Persistence.print_barcode/1" do
    test "returns :ok for a normal result" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      result = Persistence.compute(pts)
      assert :ok == Persistence.print_barcode(result)
    end

    test "returns :ok for a single-point result" do
      pts = Nx.tensor([[0.0, 0.0]])
      result = Persistence.compute(pts)
      assert :ok == Persistence.print_barcode(result)
    end
  end

  # ── Persistence.compute/2 H0 + H1 ────────────────────────────────────────

  describe "Persistence.compute/2 H0 and H1 features" do
    test "four points forming a square produce both H0 and H1 pairs" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
      result = Persistence.compute(pts, max_dim: 2)
      h0_pairs = Enum.filter(result.pairs, fn {d, _, _} -> d == 0 end)
      h1_pairs = Enum.filter(result.pairs, fn {d, _, _} -> d == 1 end)
      assert h0_pairs != []
      assert h1_pairs != []
    end
  end
end
