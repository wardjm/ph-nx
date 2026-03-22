#!/usr/bin/env mix run
# Usage:
#   mix run bench/benchmark.exs
#   mix run bench/benchmark.exs -- --points 30
#   mix run bench/benchmark.exs -- --points 100 --file path/to/cloud.txt

defmodule Bench do
  @default_file "test/fixtures/o3_50.txt"
  @warmup_runs 2
  @timed_runs 5

  def main(args) do
    args = Enum.drop_while(args, &(&1 == "--"))

    {opts, _, _} =
      OptionParser.parse(args, strict: [points: :integer, file: :string, max_dim: :integer])

    file = Keyword.get(opts, :file, @default_file)
    n = Keyword.get(opts, :points, nil)
    max_dim = Keyword.get(opts, :max_dim, 2)

    raw =
      File.read!(file)
      |> String.split("\n", trim: true)
      |> then(fn lines -> if n, do: Enum.take(lines, n), else: lines end)
      |> Enum.map(fn line -> line |> String.split("\t") |> Enum.map(&String.to_float/1) end)

    actual_n = length(raw)
    IO.puts("\n#{String.duplicate("═", 60)}")
    IO.puts("  ph-nx benchmark  |  #{actual_n} points  |  max_dim=#{max_dim}")
    IO.puts("#{String.duplicate("═", 60)}\n")

    for {label, backend} <- [{"BinaryBackend", Nx.BinaryBackend}, {"EXLA (CPU)", EXLA.Backend}] do
      try do
        points = Nx.tensor(raw, type: :f64, backend: backend)
        run_backend(label, points, max_dim)
      rescue
        e in [UndefinedFunctionError, RuntimeError] ->
          IO.puts("── #{label} ──────────────────────────────────────────")
          IO.puts("  skipped: #{Exception.message(e)}\n")
      end
    end
  end

  defp run_backend(label, points, max_dim) do
    IO.puts("── #{label} ──────────────────────────────────────────")

    # Warm-up (allows EXLA to JIT-compile)
    for _ <- 1..@warmup_runs, do: PhNx.compute(points, max_dim: max_dim)

    # Phase breakdown (single run, after warm-up)
    t0 = ts()
    dist = PhNx.Distance.euclidean(points)
    radius = PhNx.Distance.enclosing_radius(dist)
    t1 = ts()
    filt = dist |> PhNx.Filtration.build(max_dim) |> Enum.filter(&(&1.birth <= radius))
    t2 = ts()
    {bnd, _} = PhNx.BoundaryMatrix.build(filt)
    t3 = ts()
    result = PhNx.Reduction.reduce(bnd, length(filt), filt)
    t4 = ts()

    pairs = length(result.pairs)
    essential = length(result.essential)
    simplices = length(filt)

    IO.puts("  Simplices        #{simplices}")
    IO.puts("  Pairs            #{pairs}  (essential: #{essential})")
    IO.puts("")
    IO.puts("  Phase             Time")
    IO.puts("  ─────────────── ──────")
    IO.puts("  distance matrix  #{fmt(t1 - t0)}")
    IO.puts("  filtration       #{fmt(t2 - t1)}")
    IO.puts("  boundary matrix  #{fmt(t3 - t2)}")
    IO.puts("  reduction        #{fmt(t4 - t3)}")
    IO.puts("  ─────────────── ──────")
    IO.puts("  total            #{fmt(t4 - t0)}")

    # Timed runs for stable average
    times =
      for _ <- 1..@timed_runs do
        t = ts()
        PhNx.compute(points, max_dim: max_dim)
        ts() - t
      end

    avg = Enum.sum(times) / @timed_runs
    min = Enum.min(times)
    IO.puts("")
    IO.puts("  #{@timed_runs}-run avg: #{fmt(avg)}   min: #{fmt(min)}")
    IO.puts("")
  end

  defp ts, do: System.monotonic_time(:microsecond)
  defp fmt(us), do: "#{Float.round(us / 1000, 2)}ms" |> String.pad_leading(9)
end

Bench.main(System.argv())
