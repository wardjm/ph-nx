defmodule PhNx.PersistenceTest do
  use ExUnit.Case, async: true

  alias PhNx.{BoundaryMatrix, Persistence}

  @points Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

  describe "Persistence.compute/2 boundary_builder option" do
    test "passthrough builder produces identical result to the default" do
      passthrough = fn filtration, opts ->
        BoundaryMatrix.build_from_filtration(filtration, opts)
      end

      default_result = Persistence.compute(@points)
      custom_result = Persistence.compute(@points, boundary_builder: passthrough)

      assert custom_result == default_result
    end

    test "calls custom boundary_builder and its result propagates to the output" do
      test_pid = self()

      custom_builder = fn filtration, opts ->
        send(test_pid, :custom_builder_called)
        BoundaryMatrix.build_from_filtration(filtration, opts)
      end

      result = Persistence.compute(@points, boundary_builder: custom_builder)

      assert_received :custom_builder_called
      assert %{pairs: _, essential: _, diagram: _} = result
    end

    test "raises ArgumentError when boundary_builder is not a 2-arity function" do
      assert_raise ArgumentError, ~r/boundary_builder must be a 2-arity function/, fn ->
        Persistence.compute(@points, boundary_builder: :not_a_function)
      end
    end
  end

  describe "Persistence.compute_stream/2" do
    test "produces same result as compute/2 for a point cloud" do
      points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

      batch_result = Persistence.compute(points)
      stream_result = Persistence.compute_stream(points)

      assert batch_result == stream_result
    end

    test "accepts an empty stream and raises ArgumentError" do
      assert_raise ArgumentError, ~r/point cloud must be non-empty/, fn ->
        Persistence.compute_stream([])
      end
    end

    test "accepts options: max_dim" do
      points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

      result = Persistence.compute_stream(points, max_dim: 1)

      assert is_map(result)
      assert Map.has_key?(result, :pairs)
      assert Map.has_key?(result, :essential)
      assert Map.has_key?(result, :diagram)
    end

    test "accepts options: threshold" do
      points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

      result = Persistence.compute_stream(points, threshold: 0.5)

      assert is_map(result)
      assert Map.has_key?(result, :pairs)
    end

    test "accepts options: boundary_builder" do
      points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
      test_pid = self()

      custom_builder = fn filtration, opts ->
        send(test_pid, :stream_builder_called)
        BoundaryMatrix.build_from_filtration(filtration, opts)
      end

      Persistence.compute_stream(points, boundary_builder: custom_builder)

      assert_received :stream_builder_called
    end

    test "raises ArgumentError for invalid max_dim" do
      assert_raise ArgumentError, ~r/max_dim must be a non-negative integer/, fn ->
        Persistence.compute_stream([[0.0, 0.0]], max_dim: -1)
      end
    end

    test "raises ArgumentError for invalid threshold" do
      assert_raise ArgumentError, ~r/threshold must be :infinity or a non-negative number/, fn ->
        Persistence.compute_stream([[0.0, 0.0]], threshold: -1.0)
      end
    end

    test "raises ArgumentError when boundary_builder is not a function" do
      assert_raise ArgumentError, ~r/boundary_builder must be a 2-arity function/, fn ->
        Persistence.compute_stream([[0.0, 0.0]], boundary_builder: :not_a_function)
      end
    end

    test "works with 1D points" do
      points = [[0.0], [1.0], [2.0], [3.0]]

      result = Persistence.compute_stream(points)

      assert is_map(result)
      assert Map.has_key?(result, :diagram)
    end

    test "works with 3D points" do
      points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

      result = Persistence.compute_stream(points)

      assert is_map(result)
      assert Map.has_key?(result, :diagram)
    end
  end
end
