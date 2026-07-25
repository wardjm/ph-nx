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
    # --no-start so a broken accelerator install cannot abort the task before
    # the CLI runs; PhNx.Backend starts the backend itself once it has checked
    # that it is usable, and reports a fallback instead of crashing.
    Mix.Task.run("app.start", ["--no-start"])
    Application.ensure_all_started(:nx)
    PhNx.CLI.main(args)
  end
end
