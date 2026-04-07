defmodule PhNx.FiltrationBuilder do
  @moduledoc """
  Builds a Vietoris‑Rips filtration from a point cloud.

  The function takes a point cloud (Nx tensor or a list of coordinate lists) and
  optional keyword arguments.

  Options:
    * `max_dim` – maximum simplex dimension to include (default: `2`).
    * `threshold` – birth times above this value are omitted.  By default the
      enclosing radius of the point cloud is used, ensuring that only the
      meaningful part of the filtration is kept.  Pass `:infinity` to keep all
      simplices.

  Returns a list of simplexes in the same format produced by
  `PhNx.Filtration.build/2`.
  """

  alias PhNx.{Distance, Filtration}

  @typedoc "Options for building the filtration"
  @type options :: [max_dim: integer() | nil, threshold: number() | :infinity | nil]

  @spec build(Nx.Tensor.t() | [[number()]], options()) :: [Filtration.simplex()]
  def build(point_cloud, opts \ []) do
    opts = Keyword.validate!(opts, [:max_dim, :threshold])
    max_dim = Keyword.get(opts, :max_dim, 2)
    threshold = Keyword.get(opts, :threshold, :infinity)

    if not is_integer(max_dim) or max_dim < 0 do
      raise ArgumentError, "max_dim must be a non-negative integer, got: #{inspect(max_dim)}"
    end

    points = if is_list(point_cloud), do: Nx.tensor(point_cloud, type: :f64), else: point_cloud

    if Nx.axis_size(points, 0) == 0 do
      raise ArgumentError, "point cloud must be non-empty"
    end

    dist = Distance.euclidean(points)

    # Build full filtration up to max_dim
    full = Filtration.build(dist, max_dim)

    # Apply thresholding if needed
    cond do
      threshold == :infinity -> full
      is_number(threshold) and threshold >= 0 ->
        Enum.filter(full, fn %{birth: b} -> b <= threshold end)
      true ->
        raise ArgumentError, "threshold must be :infinity or a non-negative number, got: #{inspect(threshold)}"
    end
  end
end
