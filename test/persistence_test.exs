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

    test "calls custom boundary_builder when provided" do
      test_pid = self()

      custom_builder = fn filtration, _opts ->
        send(test_pid, :custom_builder_called)
        BoundaryMatrix.build_from_filtration(filtration)
      end

      Persistence.compute(@points, boundary_builder: custom_builder)

      assert_received :custom_builder_called
    end
  end
end
