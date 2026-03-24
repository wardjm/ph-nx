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

  describe "BoundaryMatrix.from_filtration/2" do
    test "vertices-only filtration has no non-zero columns" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      bm = BoundaryMatrix.from_filtration(f)
      # No edges → to_tensor gives zero matrix
      t = BoundaryMatrix.to_tensor(bm, length(f))
      assert Nx.to_list(t) == [[0, 0], [0, 0]]
    end

    test "edge columns each have exactly 2 non-zero rows" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.from_filtration(f)
      m = length(f)
      t = BoundaryMatrix.to_tensor(bm, m)

      Enum.each(Enum.filter(f, &(&1.dim == 1)), fn %{index: j} ->
        col = t[[.., j]] |> Nx.to_list()
        assert Enum.sum(col) == 2
      end)
    end

    test "triangle columns each have exactly 3 non-zero rows" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      bm = BoundaryMatrix.from_filtration(f)
      m = length(f)
      t = BoundaryMatrix.to_tensor(bm, m)

      Enum.each(Enum.filter(f, &(&1.dim == 2)), fn %{index: j} ->
        col = t[[.., j]] |> Nx.to_list()
        assert Enum.sum(col) == 3
      end)
    end

    test "seed_apparent: false produces same tensor as default (columns unchanged)" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm_default = BoundaryMatrix.from_filtration(f)
      bm_bare = BoundaryMatrix.from_filtration(f, seed_apparent: false)
      m = length(f)
      assert BoundaryMatrix.to_tensor(bm_default, m) == BoundaryMatrix.to_tensor(bm_bare, m)
    end
  end

  describe "BoundaryMatrix.lowest/2" do
    test "returns max row index in a non-zero edge column" do
      # Two-point filtration: [0] idx 0, [1] idx 1, [0,1] idx 2
      # Column 2 has boundary {0, 1} → lowest = 1
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.from_filtration(f, seed_apparent: false)
      assert BoundaryMatrix.lowest(bm, 2) == 1
    end

    test "returns nil for a vertex column (no boundary)" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.from_filtration(f, seed_apparent: false)
      assert BoundaryMatrix.lowest(bm, 0) == nil
    end

    test "returns max face index for a triangle column (3 boundary rows)" do
      # Right triangle: vertices 0,1,2 at indices 0-2; edges at 3-5; triangle at 6.
      # Triangle boundary = {3, 4, 5}; lowest = max = 5.
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      bm = BoundaryMatrix.from_filtration(f, seed_apparent: false)
      tri = Enum.find(f, &(&1.dim == 2))

      edge_indices =
        Filtration.faces(tri)
        |> Enum.map(fn verts -> Enum.find(f, &(&1.vertices == verts)).index end)

      assert BoundaryMatrix.lowest(bm, tri.index) == Enum.max(edge_indices)
    end

    test "returns nil for a column absent from the matrix" do
      pts = Nx.tensor([[0.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      bm = BoundaryMatrix.from_filtration(f)
      assert BoundaryMatrix.lowest(bm, 99) == nil
    end
  end

  describe "BoundaryMatrix.to_tensor/2" do
    test "vertices-only filtration produces zero matrix" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      bm = BoundaryMatrix.from_filtration(f)
      t = BoundaryMatrix.to_tensor(bm, 3)
      assert Nx.shape(t) == {3, 3}
      assert Nx.to_list(t) == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    end

    test "two-point filtration: edge column has rows 0 and 1 set" do
      # [0] idx 0, [1] idx 1, [0,1] idx 2 → column 2 = {0, 1}
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.from_filtration(f)
      t = BoundaryMatrix.to_tensor(bm, 3)
      mat = Nx.to_list(t)
      assert Enum.at(Enum.at(mat, 0), 2) == 1
      assert Enum.at(Enum.at(mat, 1), 2) == 1
      assert Enum.at(Enum.at(mat, 2), 2) == 0
    end

    test "dim=1 filtration: each edge column has exactly 2 ones, vertex columns are zero" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.from_filtration(f)
      m = length(f)
      t = BoundaryMatrix.to_tensor(bm, m)
      assert Nx.shape(t) == {m, m}

      Enum.each(f, fn %{index: j, dim: dim} ->
        col_sum = t[[.., j]] |> Nx.to_list() |> Enum.sum()
        expected = if dim == 0, do: 0, else: 2

        assert col_sum == expected,
               "column #{j} (dim #{dim}) expected #{expected} ones, got #{col_sum}"
      end)
    end
  end

  # ── BoundaryMatrix.reduce_column/2 ───────────────────────────────────────────

  describe "BoundaryMatrix.reduce_column/2" do
    # Two-point filtration: [0] idx 0, [1] idx 1, [0,1] idx 2
    # Bare matrix (no apparent pairs).
    defp two_point_bare do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      BoundaryMatrix.from_filtration(f, seed_apparent: false)
    end

    test ":zero — vertex column (no boundary) returns {:zero, matrix}" do
      bm = two_point_bare()
      assert {:zero, _} = BoundaryMatrix.reduce_column(bm, 0)
    end

    test ":already_resolved — column pre-seeded by apparent pair returns {:already_resolved, matrix}" do
      # Two-point filtration with apparent pairs seeded: edge [0,1] (idx 2)
      # forms an apparent pair with vertex [1] (idx 1). So col 2 is in ap_resolved.
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.from_filtration(f)
      assert {:already_resolved, _} = BoundaryMatrix.reduce_column(bm, 2)
    end

    test ":paired — edge column with unique lowest row returns {:paired, matrix}" do
      # Bare two-point filtration: v0 idx 0, v1 idx 1, e01 idx 2.
      # Column 2 has boundary {0,1}, lowest=1. No existing pivot at row 1 → paired.
      bm = two_point_bare()
      assert {:paired, _} = BoundaryMatrix.reduce_column(bm, 2)
    end

    test "clearing — birth column is zeroed out after pairing" do
      # After reducing col 2, its lowest row (1) becomes the pivot.
      # Clearing lemma: col 1 (the birth) must be deleted → lowest(bm2, 1) == nil.
      bm = two_point_bare()
      {:paired, bm2} = BoundaryMatrix.reduce_column(bm, 2)
      assert BoundaryMatrix.lowest(bm2, 1) == nil
    end

    test ":paired after XOR — column requiring prior XOR still pairs correctly" do
      # 4-point square (dim=2), seeded apparent pairs.
      # v0-v3 idx 0-3, edges idx 4-9, triangles idx 10-13.
      # Apparent pairs: {1,4},{2,5},{3,6},{9,10},{8,11} → ap_resolved includes 10,11.
      # Column 12 (t023, boundary={5,7,8}):
      #   lowest=8, pivot_col[8]=11 → XOR with col11 first, then finds lowest=7 (no pivot) → paired.
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      bm = BoundaryMatrix.from_filtration(f)
      tri023 = Enum.find(f, &(&1.vertices == [0, 2, 3]))
      assert {:paired, bm2} = BoundaryMatrix.reduce_column(bm, tri023.index)
      # The birth edge (e23, the one cleared) should now have lowest=nil.
      e23 = Enum.find(f, &(&1.vertices == [2, 3]))
      assert BoundaryMatrix.lowest(bm2, e23.index) == nil
    end
  end

  # ── BoundaryMatrix.reduce/1 and result/1 ─────────────────────────────────────

  describe "BoundaryMatrix.reduce/1 (t())" do
    test "two-point filtration: pairs and essential via result/1" do
      # Filtration: v0(0), v1(1), e01(2). Pair: {1,2}. Essential: [0].
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.from_filtration(f, seed_apparent: false)
      {pairs, essential} = bm |> BoundaryMatrix.reduce() |> BoundaryMatrix.result()
      assert {1, 2} in pairs
      assert 0 in essential
    end

    test "triangle: result/1 essential has exactly one H0 and one H1 essential class" do
      # 3-point right triangle (dim=2): v0,v1,v2 → edges → triangle fills the loop.
      # Expected: 1 essential H0 (v0), triangle kills the H1 loop → no essential H1.
      # But with seed_apparent: false the essential classes are indices: [0].
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      bm = BoundaryMatrix.from_filtration(f, seed_apparent: false)
      {pairs, essential} = bm |> BoundaryMatrix.reduce() |> BoundaryMatrix.result()
      # 2 edges kill 2 H0 classes; 1 triangle kills the H1 loop
      assert length(pairs) == 3
      # Only 1 essential class (the lone connected component, index 0)
      assert essential == [0]
    end

    test "reduce(filtration) is equivalent to from_filtration |> reduce" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)

      {pairs_via_list, essential_via_list} =
        f |> BoundaryMatrix.reduce() |> BoundaryMatrix.result()

      {pairs_via_bm, essential_via_bm} =
        f
        |> BoundaryMatrix.from_filtration()
        |> BoundaryMatrix.reduce()
        |> BoundaryMatrix.result()

      assert Enum.sort(pairs_via_list) == Enum.sort(pairs_via_bm)
      assert Enum.sort(essential_via_list) == Enum.sort(essential_via_bm)
    end

    test "matches Reduction.reduce/1 pairs and essential on triangle (seeded)" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      bm = BoundaryMatrix.from_filtration(f)
      %{pairs: r_pairs, essential: r_essential} = Reduction.reduce(bm)
      {bm_pairs, bm_essential} = bm |> BoundaryMatrix.reduce() |> BoundaryMatrix.result()
      assert Enum.sort(bm_pairs) == Enum.sort(r_pairs)
      assert Enum.sort(bm_essential) == Enum.sort(r_essential)
    end

    test "matches Reduction.reduce/1 pairs and essential on triangle (bare, no apparent pairs)" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      bm = BoundaryMatrix.from_filtration(f, seed_apparent: false)
      %{pairs: r_pairs, essential: r_essential} = Reduction.reduce(bm)
      {bm_pairs, bm_essential} = bm |> BoundaryMatrix.reduce() |> BoundaryMatrix.result()
      assert Enum.sort(bm_pairs) == Enum.sort(r_pairs)
      assert Enum.sort(bm_essential) == Enum.sort(r_essential)
    end

    test "circle (unit square): reduce(filtration) |> result() asserts H1 {dim,birth,death}" do
      # 4 points forming a unit square — the loop is born at edge length 1.0
      # and killed at the first diagonal (length sqrt(2)).
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      threshold = Distance.enclosing_radius(d)
      filtration = d |> Filtration.build(2) |> Enum.filter(&(&1.birth <= threshold))

      {raw_pairs, _essential} =
        filtration |> BoundaryMatrix.reduce() |> BoundaryMatrix.result()

      idx_map = Map.new(filtration, fn s -> {s.index, s} end)

      triples =
        raw_pairs
        |> Enum.map(fn {i, j} ->
          si = Map.fetch!(idx_map, i)
          sj = Map.fetch!(idx_map, j)
          {si.dim, si.birth, sj.birth}
        end)
        |> Enum.filter(fn {_dim, birth, death} -> birth < death end)

      h1 = Enum.filter(triples, fn {d, _, _} -> d == 1 end)
      assert length(h1) == 1
      [{1, birth, death}] = h1
      assert_in_delta birth, 1.0, 1.0e-9
      assert_in_delta death, :math.sqrt(2), 1.0e-9
    end
  end

  # ── Reduction ────────────────────────────────────────────────────────────────

  describe "Reduction.reduce/1 (with apparent pairs pre-seeded)" do
    test "produces same pairs and essential as bare-matrix reduction" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      bm_bare = BoundaryMatrix.from_filtration(f, seed_apparent: false)
      bm_seeded = BoundaryMatrix.from_filtration(f)
      r_bare = Reduction.reduce(bm_bare)
      r_seeded = Reduction.reduce(bm_seeded)
      assert Enum.sort(r_seeded.pairs) == Enum.sort(r_bare.pairs)
      assert Enum.sort(r_seeded.essential) == Enum.sort(r_bare.essential)
    end

    test "apparent pair is recorded directly in result (two-point filtration)" do
      # Filtration: v0(0), v1(1), e01(2). Apparent pair: (1, 2).
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.from_filtration(f)
      result = Reduction.reduce(bm)
      assert {1, 2} in result.pairs
      assert 0 in result.essential
    end
  end

  describe "Reduction.reduce/1" do
    test "single vertex has no pairs, one essential class" do
      pts = Nx.tensor([[0.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 0)
      bm = BoundaryMatrix.from_filtration(f)
      result = Reduction.reduce(bm)
      assert result.pairs == []
      assert result.essential == [0]
    end

    test "clearing: every birth column is absent from the reduced matrix" do
      # For any pair (i, j), column i is pre-cleared after the pair is claimed.
      # Uses seed_apparent: false so no ap-death columns are protected.
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 2)
      bm = BoundaryMatrix.from_filtration(f, seed_apparent: false)
      %{pairs: pairs, reduced: reduced} = Reduction.reduce(bm)

      Enum.each(pairs, fn {i, _j} ->
        assert BoundaryMatrix.lowest(reduced, i) == nil,
               "birth column #{i} should be cleared after pairing"
      end)
    end

    test "two isolated vertices: one edge pair kills one H0 class" do
      pts = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      d = Distance.euclidean(pts)
      f = Filtration.build(d, 1)
      bm = BoundaryMatrix.from_filtration(f)
      result = Reduction.reduce(bm)
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

      assert %{pairs: _, essential: _, diagram: _} =
               Persistence.compute(pts, max_dim: 1, threshold: 2.0)
    end

    test "identical points: zero-persistence pair is filtered, 1 essential H0 remains" do
      # The edge [0,1] has birth = 0.0 (distance is 0), pairing with vertex 1
      # at death = 0.0. Since birth == death the pair has zero persistence and
      # is filtered out by compute/2, leaving only one essential H0 class.
      pts = Nx.tensor([[0.0, 0.0], [0.0, 0.0]])
      result = Persistence.compute(pts, max_dim: 1)
      h0_essential = Enum.filter(result.essential, fn {d, _} -> d == 0 end)
      h0_pairs = Enum.filter(result.pairs, fn {d, _, _} -> d == 0 end)
      assert length(h0_essential) == 1
      assert h0_pairs == []
    end

    test "single-dimensional point cloud has 1 essential H0 and no H1 features" do
      # Three collinear points form a path graph — no loops possible.
      pts = Nx.tensor([[0.0], [1.0], [2.0]])
      result = Persistence.compute(pts, max_dim: 1)
      h0_essential = Enum.filter(result.essential, fn {d, _} -> d == 0 end)
      h1_pairs = Enum.filter(result.pairs, fn {d, _, _} -> d == 1 end)
      assert length(h0_essential) == 1
      assert h1_pairs == []
    end

    test "large max_dim does not crash" do
      # 8 collinear points with max_dim=5 exercises high-dimensional simplex
      # generation without producing any H1+ topology.
      pts = Nx.tensor(for i <- 1..8, do: [i * 1.0, 0.0])
      assert %{pairs: _, essential: _, diagram: _} = Persistence.compute(pts, max_dim: 5)
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
