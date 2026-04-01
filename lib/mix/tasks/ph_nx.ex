defmodule Mix.Tasks.PhNx do
  @shortdoc "Compute persistent homology from a point cloud file"
  @moduledoc """
  Compute persistent homology from a point cloud file.

  See `PhNx.CLI` for full usage.
  """

  use Mix.Task

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")
    PhNx.CLI.main(args)
  end
end
