defmodule PhNx.CLI do
  @moduledoc """
  Command-line interface for computing persistent homology from a point cloud file.

  ## Usage

      ph_nx <file>

  The file should contain one point per line, with coordinates comma-separated:

      0.0,0.0
      1.0,0.0
      1.0,1.0
      0.0,1.0

  Points can be in any dimension as long as all points have the same number of coordinates.

  ## Options

      --max-dim N     Maximum homology dimension to compute (default: 2)
      --threshold T   Distance threshold for filtration (default: enclosing radius)
  """

  def main(args) do
    {opts, positional, invalid} =
      OptionParser.parse(args,
        strict: [max_dim: :integer, threshold: :float, help: :boolean]
      )

    cond do
      invalid != [] ->
        IO.puts(:stderr, "Unknown option(s): #{Enum.map_join(invalid, ", ", fn {k, _} -> k end)}")
        print_usage()
        System.halt(1)

      opts[:help] ->
        print_usage()

      positional == [] ->
        IO.puts(:stderr, "Error: no input file specified")
        print_usage()
        System.halt(1)

      true ->
        [file | _] = positional
        run(file, opts)
    end
  end

  defp run(file, opts) do
    points =
      case read_points(file) do
        {:ok, pts} ->
          pts

        {:error, reason} ->
          IO.puts(:stderr, "Error reading #{file}: #{reason}")
          System.halt(1)
      end

    compute_opts =
      []
      |> maybe_put(:max_dim, opts[:max_dim])
      |> maybe_put(:threshold, opts[:threshold])

    IO.puts("Computing persistent homology for #{length(points)} points in #{dim(points)}D...")

    result = PhNx.compute(points, compute_opts)
    PhNx.print_barcode(result)

    betti = PhNx.betti_numbers(result)

    IO.puts("\nBetti numbers:")

    Enum.each(betti, fn {dim, count} ->
      IO.puts("  β#{dim} = #{count}")
    end)
  end

  defp read_points(file) do
    case File.read(file) do
      {:error, reason} ->
        {:error, :file.format_error(reason)}

      {:ok, contents} ->
        lines =
          contents
          |> String.split("\n", trim: true)
          |> Enum.reject(&(String.trim(&1) == "" or String.starts_with?(String.trim(&1), "#")))

        case parse_points(lines) do
          {:error, _} = err -> err
          points -> {:ok, points}
        end
    end
  end

  defp parse_points(lines) do
    lines
    |> Enum.with_index(1)
    |> Enum.reduce_while([], fn {line, lineno}, acc ->
      coords =
        line
        |> String.split(~r/[,\t]/)
        |> Enum.map(fn s ->
          s |> String.trim() |> Float.parse()
        end)

      if Enum.all?(coords, &match?({_, ""}, &1)) do
        {:cont, [Enum.map(coords, fn {f, _} -> f end) | acc]}
      else
        {:halt, {:error, "invalid number on line #{lineno}: #{inspect(line)}"}}
      end
    end)
    |> case do
      {:error, _} = err -> err
      points -> Enum.reverse(points)
    end
  end

  defp dim([first | _]), do: length(first)
  defp dim([]), do: 0

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)

  defp print_usage do
    IO.puts("""
    Usage: ph_nx <file> [options]

    Compute persistent homology from a point cloud file.
    Each line in the file should be a comma-separated list of coordinates:

        0.0,0.0
        1.0,0.0
        1.0,1.0
        0.0,1.0

    Options:
      --max-dim N     Maximum homology dimension to compute (default: 2)
      --threshold T   Distance threshold for filtration (default: enclosing radius)
      --help          Show this help message
    """)
  end
end
