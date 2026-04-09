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

  @doc """
  Lazily stream simplices from a Vietoris-Rips filtration without loading the
  entire complex into memory.

  Simplices are yielded dimension-by-dimension (all vertices first, then edges,
  then triangles, etc.). `max_dim` and `threshold` are applied during generation,
  so filtered simplices are never materialised.

  Unlike `build/2`, the returned simplices do **not** carry an `:index` field,
  and the stream is not globally sorted across dimensions by birth time.
  """
  @spec stream(Nx.Tensor.t() | [[number()]], options()) :: Enumerable.t()
  def stream(point_cloud, opts \\ []) do
    {max_dim, threshold} = parse_opts(opts)
    points = prepare_points(point_cloud)
    n = Nx.axis_size(points, 0)
    dist_flat = points |> Distance.euclidean() |> Distance.flat_tuple()

    0..max_dim
    |> Stream.flat_map(fn dim ->
      combinations(Enum.to_list(0..(n - 1)), dim + 1)
      |> Stream.map(fn verts ->
        birth =
          for i <- verts, j <- verts, i < j, reduce: 0.0 do
            acc -> max(acc, elem(dist_flat, i * n + j))
          end

        %{vertices: verts, dim: dim, birth: birth}
      end)
      |> Stream.filter(fn %{birth: b} -> threshold == :infinity or b <= threshold end)
    end)
  end

  @spec build(Nx.Tensor.t() | [[number()]], options()) :: [Filtration.simplex()]
  def build(point_cloud, opts \\ []) do
    {max_dim, threshold} = parse_opts(opts)
    points = prepare_points(point_cloud)
    dist = Distance.euclidean(points)
    full = Filtration.build(dist, max_dim)

    if threshold == :infinity,
      do: full,
      else: Enum.filter(full, fn %{birth: b} -> b <= threshold end)
  end

  # ---------------------------------------------------------------------------
  # Private helpers
  # ---------------------------------------------------------------------------

  defp parse_opts(opts) do
    opts = Keyword.validate!(opts, [:max_dim, :threshold])
    max_dim = Keyword.get(opts, :max_dim, 2)
    threshold = Keyword.get(opts, :threshold, :infinity)

    unless is_integer(max_dim) and max_dim >= 0 do
      raise ArgumentError, "max_dim must be a non-negative integer, got: #{inspect(max_dim)}"
    end

    unless threshold == :infinity or (is_number(threshold) and threshold >= 0) do
      raise ArgumentError,
            "threshold must be :infinity or a non-negative number, got: #{inspect(threshold)}"
    end

    {max_dim, threshold}
  end

  defp prepare_points(point_cloud) do
    points = if is_list(point_cloud), do: Nx.tensor(point_cloud, type: :f64), else: point_cloud

    if Nx.axis_size(points, 0) == 0 do
      raise ArgumentError, "point cloud must be non-empty"
    end

    points
  end

  # Generate all k-combinations of elements from a list, in lexicographic order.
  defp combinations(_list, 0), do: [[]]
  defp combinations([], _k), do: []

  defp combinations([h | t], k) do
    with_h = combinations(t, k - 1) |> Enum.map(&[h | &1])
    without_h = combinations(t, k)
    with_h ++ without_h
  end
end
