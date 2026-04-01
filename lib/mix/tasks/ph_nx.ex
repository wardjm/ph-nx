defmodule Mix.Tasks.PhNx do
  @shortdoc "Compute persistent homology from a point cloud file"
  @moduledoc """
  Compute persistent homology from a point cloud file.

  ## Usage

      mix ph_nx <file> [options]

  The file should contain one point per line, with coordinates separated by
  commas or tabs:

      mix ph_nx points.csv
      mix ph_nx points.csv --max-dim 1
      mix ph_nx points.csv --threshold 2.5

  ## Options

      --max-dim N     Maximum homology dimension to compute (default: 2)
      --threshold T   Distance threshold for filtration (default: enclosing radius)
      --help          Show this help message
  """

  use Mix.Task

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")
    PhNx.CLI.main(args)
  end
end
