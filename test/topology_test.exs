defmodule PhNx.TopologyTest do
  use ExUnit.Case, async: true

  # Four corners of a unit square — H0: 1 component, H1: 1 loop
  @square Nx.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

  describe "PhNx.Topology.compute/2" do
    test "returns result map with pairs, essential, and diagram keys" do
      result = PhNx.Topology.compute(@square)
      assert is_map(result)
      assert Map.has_key?(result, :pairs)
      assert Map.has_key?(result, :essential)
      assert Map.has_key?(result, :diagram)
    end

    test "detects correct topology for unit square (H0: 1 component, H1: 1 loop)" do
      result = PhNx.Topology.compute(@square)

      # H0: 1 essential class (1 connected component that persists forever)
      h0_essential = Enum.filter(result.essential, fn {d, _} -> d == 0 end)
      assert length(h0_essential) == 1

      # H1: 1 finite pair (the loop)
      h1_pairs = Enum.filter(result.pairs, fn {d, _, _} -> d == 1 end)
      assert length(h1_pairs) == 1
    end

    test "matches Persistence.compute/2 output for the same input" do
      result_topology = PhNx.Topology.compute(@square)
      result_persistence = PhNx.Persistence.compute(@square)

      assert result_topology.pairs == result_persistence.pairs
      assert result_topology.essential == result_persistence.essential
      assert result_topology.diagram == result_persistence.diagram
    end

    test "respects :max_dim option" do
      result_dim1 = PhNx.Topology.compute(@square, max_dim: 1)
      result_dim2 = PhNx.Topology.compute(@square, max_dim: 2)

      # With max_dim: 1 there are no triangles to kill the H1 loop, so it
      # shows up as an essential class (infinite bar) rather than a finite pair.
      h1_essential_dim1 = Enum.filter(result_dim1.essential, fn {d, _} -> d == 1 end)
      assert length(h1_essential_dim1) >= 1

      # With max_dim: 2 triangles fill in the loop, producing a finite H1 pair.
      h1_pairs_dim2 = Enum.filter(result_dim2.pairs, fn {d, _, _} -> d == 1 end)
      assert length(h1_pairs_dim2) == 1
    end

    test "respects :threshold option" do
      # With a tiny threshold, only the closest points are included
      result = PhNx.Topology.compute(@square, threshold: 0.5)
      # At threshold 0.5, no edges of the unit square (length 1.0) are included
      # so we get 4 isolated components
      h0_essential = Enum.filter(result.essential, fn {d, _} -> d == 0 end)
      assert length(h0_essential) == 4
    end

    test "raises ArgumentError for empty point cloud" do
      assert_raise ArgumentError, fn ->
        PhNx.Topology.compute([])
      end
    end

    test "raises ArgumentError for invalid :max_dim" do
      assert_raise ArgumentError, fn ->
        PhNx.Topology.compute(@square, max_dim: -1)
      end
    end

    test "raises ArgumentError for invalid :threshold" do
      assert_raise ArgumentError, fn ->
        PhNx.Topology.compute(@square, threshold: -1.0)
      end
    end

    test "accepts custom :boundary_builder and passes opts through" do
      test_pid = self()

      custom_builder = fn filtration, opts ->
        send(test_pid, {:builder_called, length(filtration), opts})
        PhNx.BoundaryMatrix.build_from_filtration(filtration, opts)
      end

      result = PhNx.Topology.compute(@square, boundary_builder: custom_builder)

      assert_received {:builder_called, _, received_opts}
      refute Keyword.has_key?(received_opts, :boundary_builder)
      assert Map.has_key?(result, :pairs)
      assert Map.has_key?(result, :essential)
      assert Map.has_key?(result, :diagram)
    end
  end

  describe "PhNx.Topology.compute/2 reduction options" do
    test ":on_progress is invoked once per reduction column" do
      test_pid = self()

      PhNx.compute(@square, on_progress: fn progress -> send(test_pid, {:progress, progress}) end)

      assert_received {:progress, %{current: 0, total: total}}

      seen =
        for _ <- 1..(total - 1) do
          assert_received {:progress, %{current: c, total: ^total}}
          c
        end

      assert seen == Enum.to_list(1..(total - 1))
      refute_received {:progress, _}
    end

    test ":coeff reaches the boundary matrix, which reduces over Zp" do
      test_pid = self()

      spy = fn filtration, opts ->
        bm = PhNx.BoundaryMatrix.build_from_filtration(filtration, opts)
        send(test_pid, {:ring, bm.coeff_ring})
        bm
      end

      PhNx.compute(@square, coeff: {:zp, 3}, boundary_builder: spy)
      assert_received {:ring, {:zp, 3}}

      PhNx.compute(@square, boundary_builder: spy)
      assert_received {:ring, :z2}
    end

    test "Zp reduction agrees with the default Z2 reduction on the square" do
      assert PhNx.compute(@square, coeff: {:zp, 3}) == PhNx.compute(@square)
    end

    test "results are unchanged when :on_progress is passed" do
      assert PhNx.compute(@square, on_progress: fn _ -> :ok end) == PhNx.compute(@square)
    end
  end

  describe "PhNx.compute/2 public API" do
    test "still works and returns correct result (no breaking change)" do
      result = PhNx.compute(@square)
      assert Map.has_key?(result, :pairs)
      assert Map.has_key?(result, :essential)
      assert Map.has_key?(result, :diagram)
    end

    test "matches Topology.compute/2 output" do
      result_facade = PhNx.compute(@square)
      result_topology = PhNx.Topology.compute(@square)

      assert result_facade.pairs == result_topology.pairs
      assert result_facade.essential == result_topology.essential
      assert result_facade.diagram == result_topology.diagram
    end
  end
end
