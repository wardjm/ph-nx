defmodule PhNx.FiltrationBuilderTest do
  use ExUnit.Case

  alias PhNx.FiltrationBuilder

  @tri [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0]
  ]

  # Unit right-angle corner in 3D: 3 axis-aligned edges (dist=1.0), 3 cross-axis edges (dist=√2≈1.414)
  @tet [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ]

  test "build full filtration without threshold" do
    filtration = FiltrationBuilder.build(@tri, max_dim: 2)
    assert length(filtration) == 7
    vertex_0 = Enum.find(filtration, fn s -> s.dim == 0 and s.vertices == [0] end)
    assert vertex_0.birth == 0.0
    e01 = Enum.find(filtration, fn s -> s.vertices == [0, 1] end)
    assert e01.birth == 1.0
    e02 = Enum.find(filtration, fn s -> s.vertices == [0, 2] end)
    assert e02.birth == 1.0
    e12 = Enum.find(filtration, fn s -> s.vertices == [1, 2] end)
    assert e12.birth > 1.4 and e12.birth < 1.5
  end

  test "threshold filters simplices beyond birth value" do
    filtration = FiltrationBuilder.build(@tri, max_dim: 2, threshold: 1.0)
    assert Enum.all?(filtration, fn s -> s.birth <= 1.0 end)
    assert length(filtration) == 5
    assert Enum.any?(filtration, fn s -> not (s.vertices == [1, 2]) end)
    assert Enum.any?(filtration, fn s -> not (s.vertices == [0, 1, 2]) end)
  end

  test "threshold :infinity returns full filtration" do
    full = FiltrationBuilder.build(@tri, max_dim: 2)
    full_inf = FiltrationBuilder.build(@tri, max_dim: 2, threshold: :infinity)
    assert full == full_inf
  end

  test "max_dim 1 returns only vertices and edges" do
    filt = FiltrationBuilder.build(@tri, max_dim: 1)
    assert length(filt) == 6
    assert Enum.all?(filt, fn s -> s.dim <= 1 end)
  end

  # 3D point cloud tests

  test "3D tetrahedron builds a filtration with correct simplex count" do
    # max_dim=3: 4 vertices + 6 edges + 4 triangles + 1 tetrahedron = 15
    filt = FiltrationBuilder.build(@tet, max_dim: 3)
    assert length(filt) == 15
  end

  test "3D max_dim=1 returns only vertices and edges" do
    # C(4,1) = 4 vertices, C(4,2) = 6 edges
    filt = FiltrationBuilder.build(@tet, max_dim: 1)
    assert length(filt) == 10
    assert Enum.all?(filt, fn s -> s.dim <= 1 end)
  end

  test "3D max_dim=0 returns only vertices" do
    filt = FiltrationBuilder.build(@tet, max_dim: 0)
    assert length(filt) == 4
    assert Enum.all?(filt, fn s -> s.dim == 0 end)
    assert Enum.all?(filt, fn s -> s.birth == 0.0 end)
  end

  test "3D threshold=1.0 keeps only vertices and axis-aligned edges" do
    # Axis-aligned pairs (dist=1.0): {0,1}, {0,2}, {0,3} — birth ≤ 1.0
    # Cross-axis pairs (dist=√2≈1.414): {1,2}, {1,3}, {2,3} — birth > 1.0, pruned
    filt = FiltrationBuilder.build(@tet, max_dim: 3, threshold: 1.0)
    assert Enum.all?(filt, fn s -> s.birth <= 1.0 end)
    # 4 vertices + 3 axis-aligned edges
    assert length(filt) == 7

    surviving_edges = filt |> Enum.filter(&(&1.dim == 1)) |> Enum.map(& &1.vertices)
    assert surviving_edges == [[0, 1], [0, 2], [0, 3]]
  end

  test "3D max_dim=2 includes triangles but not tetrahedron" do
    # C(4,1) + C(4,2) + C(4,3) = 4 + 6 + 4 = 14
    filt = FiltrationBuilder.build(@tet, max_dim: 2)
    assert length(filt) == 14
    assert Enum.all?(filt, fn s -> s.dim <= 2 end)
  end

  test "every simplex has required Filtration struct fields" do
    filt = FiltrationBuilder.build(@tet, max_dim: 3)

    for s <- filt do
      assert is_list(s.vertices), "vertices must be a list"
      assert is_integer(s.dim) and s.dim >= 0, "dim must be a non-negative integer"
      assert is_float(s.birth) and s.birth >= 0.0, "birth must be a non-negative float"
      assert is_integer(s.index) and s.index >= 0, "index must be a non-negative integer"
      assert length(s.vertices) == s.dim + 1, "vertices count must equal dim + 1"
    end
  end

  test "3D Nx.Tensor input produces same filtration as list input" do
    list_result = FiltrationBuilder.build(@tet, max_dim: 3)
    tensor_result = FiltrationBuilder.build(Nx.tensor(@tet, type: :f64), max_dim: 3)
    assert list_result == tensor_result
  end

  test "3D filtration is sorted by birth time, dimension, then vertices" do
    filt = FiltrationBuilder.build(@tet, max_dim: 3)
    sort_keys = Enum.map(filt, fn s -> {s.birth, s.dim, s.vertices} end)
    assert sort_keys == Enum.sort(sort_keys)
  end

  test "3D filtration has contiguous 0-based indices" do
    filt = FiltrationBuilder.build(@tet, max_dim: 3)
    indices = Enum.map(filt, fn s -> s.index end)
    assert indices == Enum.to_list(0..(length(filt) - 1))
  end
end
