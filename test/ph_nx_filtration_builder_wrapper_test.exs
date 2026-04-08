defmodule PhNx.FiltrationBuilderWrapperTest do
  use ExUnit.Case, async: true

  alias PhNx.FiltrationBuilder

  @tri [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0]
  ]

  @square [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
  ]

  describe "PhNx.filtration_builder/1" do
    test "exists and is exported" do
      # Test that the function exists
      assert Code.ensure_loaded?(PhNx)
    end

    test "returns a filtration for a simple point cloud" do
      filtration = PhNx.filtration_builder(@tri)
      assert is_list(filtration)
      assert length(filtration) > 0
      assert Enum.all?(filtration, fn s -> is_map(s) and Map.has_key?(s, :birth) end)
    end

    test "delegates to FiltrationBuilder.build/2 with default options" do
      wrapper_result = PhNx.filtration_builder(@tri)
      direct_result = FiltrationBuilder.build(@tri)
      assert wrapper_result == direct_result
    end
  end

  describe "PhNx.filtration_builder/2" do
    test "accepts options keyword list" do
      filtration = PhNx.filtration_builder(@tri, max_dim: 1)
      assert is_list(filtration)
      assert Enum.all?(filtration, fn s -> s.dim <= 1 end)
    end

    test "forward max_dim option correctly" do
      # max_dim: 0 should only return vertices (3 vertices)
      wrapper_result = PhNx.filtration_builder(@tri, max_dim: 0)
      direct_result = FiltrationBuilder.build(@tri, max_dim: 0)
      assert wrapper_result == direct_result
      assert length(wrapper_result) == 3
    end

    test "forward threshold option correctly" do
      wrapper_result = PhNx.filtration_builder(@tri, threshold: 1.0)
      direct_result = FiltrationBuilder.build(@tri, threshold: 1.0)
      assert wrapper_result == direct_result
      assert Enum.all?(wrapper_result, fn s -> s.birth <= 1.0 end)
    end

    test "forward multiple options" do
      wrapper_result = PhNx.filtration_builder(@tri, max_dim: 1, threshold: 1.0)
      direct_result = FiltrationBuilder.build(@tri, max_dim: 1, threshold: 1.0)
      assert wrapper_result == direct_result
    end
  end

  describe "PhNx.filtration_builder/3" do
    test "accepts Nx.Tensor as first argument" do
      points = Nx.tensor(@tri)
      filtration = PhNx.filtration_builder(points, [])
      assert is_list(filtration)
      assert length(filtration) > 0
    end

    test "accepts Nx.Tensor with options" do
      points = Nx.tensor(@tri)
      wrapper_result = PhNx.filtration_builder(points, max_dim: 1)
      direct_result = FiltrationBuilder.build(points, max_dim: 1)
      assert wrapper_result == direct_result
    end

    test "Nx.Tensor and list produce same result" do
      list_points = @tri
      tensor_points = Nx.tensor(@tri)
      list_result = PhNx.filtration_builder(list_points)
      tensor_result = PhNx.filtration_builder(tensor_points)
      assert list_result == tensor_result
    end
  end

  describe "error handling" do
    test "raises on empty point cloud" do
      # Empty lists cause an error when building Nx.tensor
      assert_raise RuntimeError, "cannot build empty tensor", fn ->
        PhNx.filtration_builder([])
      end
    end

    test "raises on invalid max_dim" do
      assert_raise ArgumentError, ~r/max_dim must be a non-negative integer/, fn ->
        PhNx.filtration_builder(@tri, max_dim: -1)
      end
    end

    test "raises on invalid threshold" do
      assert_raise ArgumentError, ~r/threshold must be :infinity or a non-negative number/, fn ->
        PhNx.filtration_builder(@tri, threshold: -1)
      end
    end

    test "raises on unknown option" do
      assert_raise ArgumentError, fn ->
        PhNx.filtration_builder(@tri, unknown_opt: 1)
      end
    end
  end

  describe "integration with unit square" do
    test "computes full filtration for unit square" do
      filtration = PhNx.filtration_builder(@square)
      assert is_list(filtration)

      # Should have vertices, edges, and 2-simplices
      vertices = Enum.filter(filtration, fn s -> s.dim == 0 end)
      edges = Enum.filter(filtration, fn s -> s.dim == 1 end)
      faces = Enum.filter(filtration, fn s -> s.dim == 2 end)

      assert length(vertices) == 4
      assert length(edges) > 0
      assert length(faces) > 0
    end

    test "unit square with max_dim 1 for TDA pipeline" do
      filtration = PhNx.filtration_builder(@square, max_dim: 1)

      # Only vertices and edges
      assert Enum.all?(filtration, fn s -> s.dim <= 1 end)

      # Should have 4 vertices and 6 edges (4 sides + 2 diagonals)
      vertices = Enum.filter(filtration, fn s -> s.dim == 0 end)
      edges = Enum.filter(filtration, fn s -> s.dim == 1 end)

      assert length(vertices) == 4
      assert length(edges) == 6
    end
  end

  describe "backward compatibility" do
    test "existing PhNx.compute/1 still works" do
      result = PhNx.compute(@tri)
      assert is_map(result)
      assert Map.has_key?(result, :diagram)
    end

    test "existing PhNx.compute/2 still works" do
      result = PhNx.compute(@tri, [])
      assert is_map(result)
      assert Map.has_key?(result, :diagram)
    end

    test "existing PhNx.print_barcode/1 still works" do
      result = PhNx.compute(@tri)
      assert PhNx.print_barcode(result) == :ok
    end

    test "existing PhNx.most_persistent/1 still works" do
      result = PhNx.compute(@tri)
      assert is_list(PhNx.most_persistent(result))
    end

    test "existing PhNx.most_persistent/2 still works" do
      result = PhNx.compute(@tri)
      assert is_list(PhNx.most_persistent(result, 5))
    end

    test "existing PhNx.betti_numbers/1 still works" do
      result = PhNx.compute(@tri)
      assert is_map(PhNx.betti_numbers(result))
    end
  end
end
