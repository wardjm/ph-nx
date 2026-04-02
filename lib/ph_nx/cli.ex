defmodule PhNx.CLI do
  @moduledoc """
  Command-line interface for computing persistent homology from a point cloud file.

  ## Usage

      ph_nx <file> [options]

  The file should contain one point per line, with coordinates comma-separated:

      0.0,0.0
      1.0,0.0
      1.0,1.0
      0.0,1.0

  Points can be in any dimension as long as all points have the same number of coordinates.

  ## Options

      --max-dim N     Maximum homology dimension to compute (default: 2)
      --threshold T   Distance threshold for filtration (default: enclosing radius)
      --help          Show this help message

  ## Building the escript

      mix escript.build
      ./ph_nx <file>
  """

  def main(args) do
    {opts, positional, invalid} =
      OptionParser.parse(args,
        strict: [max_dim: :integer, threshold: :float, help: :boolean]
      )

    cond do
      invalid != [] ->
        IO.puts(
          :stderr,
          "Invalid option(s): #{Enum.map_join(invalid, ", ", fn
            {k, nil} -> k
            {k, v} -> "#{k}=#{v}"
          end)}"
        )

        print_usage(:stderr)
        exit({:shutdown, 1})

      opts[:help] ->
        print_usage(:stdio)
        exit({:shutdown, 0})

      positional == [] ->
        IO.puts(:stderr, "Error: no input file specified")
        print_usage(:stderr)
        exit({:shutdown, 1})

      length(positional) > 1 ->
        IO.puts(
          :stderr,
          "Error: too many arguments (expected one file, got #{length(positional)})"
        )

        print_usage(:stderr)
        exit({:shutdown, 1})

      true ->
        [file] = positional
        run(file, opts)
    end
  end

  defp run(file, opts) do
    points =
      case read_points(file) do
        {:ok, pts} ->
          pts

        {:error, :file, reason} ->
          IO.puts(:stderr, "Error reading #{file}: #{reason}")
          exit({:shutdown, 1})

        {:error, :format, reason} ->
          IO.puts(:stderr, "Error: #{reason}")
          exit({:shutdown, 2})
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

    Enum.each(betti, fn {d, count} ->
      IO.puts("  β#{d} = #{count}")
    end)
  end

  defp read_points(file) do
    case File.read(file) do
      {:error, reason} ->
        {:error, :file, :file.format_error(reason)}

      {:ok, contents} ->
        lines =
          contents
          |> String.split("\n", trim: true)
          |> Enum.reject(&(String.trim(&1) == "" or String.starts_with?(String.trim(&1), "#")))

        if lines == [] do
          {:error, :format, "file contains no data points"}
        else
          case parse_points(lines) do
            {:error, reason} -> {:error, :format, reason}
            points -> {:ok, points}
          end
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

  defp print_usage(device) do
    IO.puts(device, """
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
