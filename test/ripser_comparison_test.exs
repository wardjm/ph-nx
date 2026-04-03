defmodule PhNx.RipserComparisonTest do
  use ExUnit.Case, async: false

  alias PhNx.BoundaryMatrix

  @tolerance 1.0e-4

  # Equilateral triangle — topology is fully determined analytically:
  #   H0: 1 essential class born at 0.0 (one connected component)
  #   H1: 0 finite pairs in output — the raw reduction produces one pair, but birth == death
  #       (loop is born when the 3rd edge is added, 2-simplex fills it at the same value),
  #       so the pipeline's birth < death filter removes it as degenerate
  @triangle Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])

  # ── Helpers ─────────────────────────────────────────────────────────────────

  defp load_ripser_output(path) do
    path |> File.read!() |> parse_ripser_output()
  end

  # Returns %{dim => [{birth, death | :infinity}]} for all dims in output.
  defp parse_ripser_output(output) do
    output
    |> String.split("\n")
    |> Enum.reduce({nil, %{}}, &parse_line/2)
    |> elem(1)
  end

  defp parse_line(line, {dim, acc}) do
    case Regex.run(~r/persistence intervals in dim (\d+):/, line) do
      [_, d] ->
        {String.to_integer(d), acc}

      nil ->
        case {dim, Regex.run(~r/^\s*\[([^,]+),([^\)]*)\)/, line)} do
          {nil, _} ->
            {dim, acc}

          {_, nil} ->
            {dim, acc}

          {_, [_, b, d]} ->
            birth = parse_num(b)
            death = parse_death(d)
            {dim, Map.update(acc, dim, [{birth, death}], &(&1 ++ [{birth, death}]))}
        end
    end
  end

  defp parse_num(s) do
    s = String.trim(s)
    if String.contains?(s, "."), do: String.to_float(s), else: String.to_integer(s) * 1.0
  end

  defp parse_death(s) do
    s = String.trim(s)
    if s == "", do: :infinity, else: parse_num(s)
  end

  defp load_points(path) do
    File.read!(path)
    |> String.split("\n", trim: true)
    |> Enum.map(fn line -> line |> String.split("\t") |> Enum.map(&String.to_float/1) end)
    |> Nx.tensor(type: :f64)
  end

  # Convert PhNx result to %{dim => [{birth, death | :infinity}]}.
  defp format_our_result(result) do
    finite =
      Enum.group_by(result.pairs, fn {d, _, _} -> d end, fn {_, b, death} -> {b, death} end)

    essential =
      Enum.group_by(result.essential, fn {d, _} -> d end, fn {_, b} -> {b, :infinity} end)

    Map.merge(essential, finite, fn _k, ess, fin -> ess ++ fin end)
  end

  defp assert_dims_match(ripser_by_dim, ours_by_dim, hom_dim, label) do
    for dim <- 0..hom_dim do
      expected = Map.get(ripser_by_dim, dim, []) |> Enum.sort()
      actual = Map.get(ours_by_dim, dim, []) |> Enum.sort()

      assert length(actual) == length(expected),
             "#{label} H#{dim}: pair count mismatch — ripser=#{length(expected)}, ours=#{length(actual)}"

      Enum.zip(expected, actual)
      |> Enum.with_index()
      |> Enum.each(fn {{{eb, ed}, {ab, ad}}, i} ->
        assert_in_delta ab,
                        eb,
                        @tolerance,
                        "#{label} H#{dim}[#{i}]: birth mismatch (ripser=#{eb}, ours=#{ab})"

        case ed do
          :infinity ->
            assert ad == :infinity,
                   "#{label} H#{dim}[#{i}]: expected infinite bar, got #{ad}"

          _ ->
            assert_in_delta ad,
                            ed,
                            @tolerance,
                            "#{label} H#{dim}[#{i}]: death mismatch (ripser=#{ed}, ours=#{ad})"
        end
      end)
    end
  end

  # ── Pipeline isolation tests ─────────────────────────────────────────────────

  describe "pipeline isolation via boundary_builder:" do
    test "spy: builder is called and receives the filtration" do
      test_pid = self()

      spy = fn filtration, opts ->
        send(test_pid, {:called, length(filtration)})
        BoundaryMatrix.build_from_filtration(filtration, opts)
      end

      PhNx.compute(@triangle, max_dim: 2, boundary_builder: spy)

      assert_received {:called, size}
      refute_received {:called, _}
      assert size > 0
    end

    test "stub bypassing apparent-pairs pre-pass produces identical output to default" do
      stub = fn filtration, opts ->
        BoundaryMatrix.build_from_filtration(filtration, Keyword.put(opts, :seed_apparent, false))
      end

      default = PhNx.compute(@triangle, max_dim: 2)
      stubbed = PhNx.compute(@triangle, max_dim: 2, boundary_builder: stub)

      assert stubbed == default
    end
  end

  # ── Known-topology integration tests via boundary_builder: ───────────────────
  # These use boundary_builder: to confirm the pipeline wires the builder result
  # through the full orchestration path (index mapping, filtering, sorting).
  # The reduction algorithm runs normally; correctness is verified against
  # analytically known topology for the equilateral triangle.

  describe "known-topology assertions via boundary_builder:" do
    test "pipeline maps filtration indices to correct H0 essential class" do
      recording_builder = fn filtration, opts ->
        BoundaryMatrix.build_from_filtration(filtration, opts)
      end

      result = PhNx.compute(@triangle, max_dim: 2, boundary_builder: recording_builder)

      h0_essentials = Enum.filter(result.essential, fn {dim, _birth} -> dim == 0 end)
      assert length(h0_essentials) == 1
      {0, birth} = hd(h0_essentials)
      assert_in_delta birth, 0.0, 1.0e-10
    end

    test "pipeline filters degenerate pairs: triangle's H1 loop (birth == death) is excluded" do
      # In a VR filtration on an equilateral triangle, the H1 cycle is born and killed
      # at the same filtration value (when the 3rd edge and the 2-simplex are both added).
      # The pipeline's birth < death filter must remove such degenerate pairs.
      recording_builder = fn filtration, opts ->
        BoundaryMatrix.build_from_filtration(filtration, opts)
      end

      result = PhNx.compute(@triangle, max_dim: 2, boundary_builder: recording_builder)

      h1_pairs = Enum.filter(result.pairs, fn {dim, _b, _d} -> dim == 1 end)
      assert h1_pairs == []
    end
  end

  # ── Ripser comparison tests ───────────────────────────────────────────────────
  #
  # Each entry: {label, points_file, ripser_output_file, hom_dim}
  # ripser output was captured via: ripser --format point-cloud --dim <hom_dim> <points_file>
  # hom_dim = max homology dimension (our max_dim = hom_dim + 1, the max *simplex* dim)

  @fixtures [
    {"o3_20 H0+H1", "test/fixtures/o3_20.txt", "test/fixtures/o3_20_ripser_dim1.txt", 1},
    {"o3_30 H0+H1", "test/fixtures/o3_30.txt", "test/fixtures/o3_30_ripser_dim1.txt", 1},
    {"o3_30b H0+H1", "test/fixtures/o3_30b.txt", "test/fixtures/o3_30b_ripser_dim1.txt", 1},
    {"o3_40 H0+H1", "test/fixtures/o3_40.txt", "test/fixtures/o3_40_ripser_dim1.txt", 1},
    {"o3_50 H0+H1", "test/fixtures/o3_50.txt", "test/fixtures/o3_50_ripser_dim1.txt", 1},
    {"o3_50b H0+H1", "test/fixtures/o3_50b.txt", "test/fixtures/o3_50b_ripser_dim1.txt", 1}
  ]

  for {label, points_file, ripser_file, hom_dim} <- @fixtures do
    @label label
    @points_file points_file
    @ripser_file ripser_file
    @hom_dim hom_dim

    test "#{label}: matches ripser within #{1.0e-4}" do
      ripser = load_ripser_output(@ripser_file)
      pts = load_points(@points_file)
      # Our max_dim is max simplex dimension = hom_dim + 1
      result = PhNx.compute(pts, max_dim: @hom_dim + 1)
      ours = format_our_result(result)
      assert_dims_match(ripser, ours, @hom_dim, @label)
    end
  end
end
