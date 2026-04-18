defmodule PhNx.ComputeBoundaryIntegrationTest do
  use ExUnit.Case, async: true

  alias PhNx.BoundaryMatrix

  @two_points [[0.0, 0.0], [1.0, 0.0]]
  @square [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

  # Delegates to the real builder while recording the call for assertion.
  defp spy_builder(test_pid) do
    fn filtration, opts ->
      send(test_pid, {:builder_called, filtration, opts})
      BoundaryMatrix.build_from_filtration(filtration, opts)
    end
  end

  describe "PhNx.compute/2 boundary integration" do
    test "invokes the injected boundary_builder with a non-empty filtration" do
      result = PhNx.compute(@two_points, boundary_builder: spy_builder(self()))
      assert_received {:builder_called, filtration, _opts}
      assert is_list(filtration) and filtration != []
      assert is_map(result)
    end

    test "strips :boundary_builder from opts passed to the builder" do
      PhNx.compute(@two_points, boundary_builder: spy_builder(self()))
      assert_received {:builder_called, _filtration, opts}
      refute Keyword.has_key?(opts, :boundary_builder)
    end

    test "forwards other opts (e.g. max_dim) through to the builder" do
      PhNx.compute(@two_points, boundary_builder: spy_builder(self()), max_dim: 2)
      assert_received {:builder_called, _filtration, opts}
      assert Keyword.get(opts, :max_dim) == 2
    end

    test "raises ArgumentError for a non-2-arity boundary_builder" do
      assert_raise ArgumentError, fn ->
        PhNx.compute(@two_points, boundary_builder: :bad)
      end
    end

    test "maps raw boundary pairs to {dim, birth, death} triples" do
      # Two points at distance 1.0: one H0 pair born 0.0, killed at 1.0
      result = PhNx.compute(@two_points, boundary_builder: spy_builder(self()))
      assert [{0, birth, death}] = result.pairs
      assert_in_delta birth, 0.0, 1.0e-9
      assert_in_delta death, 1.0, 1.0e-9
    end

    test "maps raw essential indices to {dim, birth} pairs" do
      # Two points: one essential H0 class (the surviving component, born at 0.0)
      result = PhNx.compute(@two_points, boundary_builder: spy_builder(self()))
      assert [{0, birth}] = result.essential
      assert_in_delta birth, 0.0, 1.0e-9
    end

    test "filters out zero-persistence pairs where birth == death" do
      # Two identical points: edge birth == 0.0, vertex birth == 0.0 → birth == death → filtered
      result = PhNx.compute([[0.0, 0.0], [0.0, 0.0]], boundary_builder: spy_builder(self()))
      assert result.pairs == []
    end

    test "diagram is the union of finite pairs and essential classes extended to :infinity" do
      # Square: 3 H0 finite pairs + 1 H1 finite pair + 1 H0 essential + 1 H1 essential
      result = PhNx.compute(@square, boundary_builder: spy_builder(self()))
      assert length(result.pairs) == 4
      assert length(result.essential) == 2
      expected_count = length(result.pairs) + length(result.essential)
      assert length(result.diagram) == expected_count
      essential_in_diagram = Enum.map(result.essential, fn {d, b} -> {d, b, :infinity} end)
      assert Enum.all?(essential_in_diagram, &(&1 in result.diagram))
      assert Enum.all?(result.pairs, &(&1 in result.diagram))
    end
  end
end
