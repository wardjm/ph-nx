defmodule PhNx.Topology do
  @moduledoc """
  Façade that orchestrates the persistent homology pipeline.

  Delegates to the three deep modules in sequence:

      Distance  →  FiltrationBuilder  →  Reduction  →  Persistence pairs

  Preserves the same `compute/2` signature as `PhNx.Persistence`.
  """

  alias PhNx.{Distance, FiltrationBuilder, Reduction, BoundaryMatrix}

  @typedoc "Options accepted by `compute/2`."
  @type options :: [
          max_dim: non_neg_integer(),
          threshold: number() | :infinity,
          boundary_builder: (list(), keyword() -> BoundaryMatrix.t()),
          backend: :cpu | :gpu,
          coeff: :z2 | {:zp, pos_integer()},
          on_progress: (%{current: non_neg_integer(), total: pos_integer()} -> any())
        ]

  @doc """
  Compute persistent homology for a point cloud.

  Options:
    - `:max_dim` (non-negative integer, default `2`) — maximum simplex dimension.
    - `:threshold` (non-negative number or `:infinity`, default: enclosing radius) —
      ignore simplices born after this value.
    - `:boundary_builder` (`(filtration, opts -> BoundaryMatrix.t())`, default:
      `&BoundaryMatrix.build_from_filtration/2`) — override the boundary matrix
      constructor (useful for testing and alternative implementations).
    - `:backend` (`:cpu` or `:gpu`, default `:cpu`) — distance computation backend.
      `:gpu` requires EXLA with a CUDA-capable device; falls back to CPU with a
      warning if the GPU is unavailable. Can also be set globally via
      `config :ph_nx, distance_backend: :gpu`.
    - `:coeff` — coefficient ring; `{:zp, p}` for ℤₚ arithmetic (default: ℤ₂).
    - `:on_progress` — 1-arity callback invoked once per column during reduction,
      receiving `%{current: non_neg_integer(), total: pos_integer()}`.

  Returns `%{pairs: [...], essential: [...], diagram: [...]}`.
  """
  @spec compute(Nx.Tensor.t() | [[number()]], options()) :: PhNx.Persistence.result()
  def compute(points, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :max_dim,
        :threshold,
        :boundary_builder,
        :backend,
        :coeff,
        :on_progress
      ])

    max_dim = Keyword.get(opts, :max_dim, 2)

    unless is_integer(max_dim) and max_dim >= 0 do
      raise ArgumentError, "max_dim must be a non-negative integer, got: #{inspect(max_dim)}"
    end

    if points == [] do
      raise ArgumentError, "point cloud must be non-empty"
    end

    points = if is_list(points), do: Nx.tensor(points, type: :f64), else: points

    if Nx.axis_size(points, 0) == 0 do
      raise ArgumentError, "point cloud must be non-empty"
    end

    dist_opts = Keyword.take(opts, [:backend])
    dist = Distance.euclidean(points, dist_opts)

    threshold =
      case Keyword.fetch(opts, :threshold) do
        {:ok, t} -> t
        :error -> Distance.enclosing_radius(dist)
      end

    unless threshold == :infinity or (is_number(threshold) and threshold >= 0) do
      raise ArgumentError,
            "threshold must be :infinity or a non-negative number, got: #{inspect(threshold)}"
    end

    filtration = FiltrationBuilder.build(points, max_dim: max_dim, threshold: threshold)

    builder = Keyword.get(opts, :boundary_builder, &BoundaryMatrix.build_from_filtration/2)

    unless is_function(builder, 2) do
      raise ArgumentError,
            "boundary_builder must be a 2-arity function, got: #{inspect(builder)}"
    end

    builder_opts = Keyword.delete(opts, :boundary_builder)
    reduced = builder.(filtration, builder_opts) |> Reduction.reduce(builder_opts)
    raw_pairs = BoundaryMatrix.pairs(reduced)
    raw_essential = BoundaryMatrix.essential(reduced)

    idx_to_simplex = Map.new(filtration, fn s -> {s.index, s} end)

    pairs =
      raw_pairs
      |> Enum.map(fn {i, j} ->
        s_i = Map.fetch!(idx_to_simplex, i)
        s_j = Map.fetch!(idx_to_simplex, j)
        {s_i.dim, s_i.birth, s_j.birth}
      end)
      |> Enum.filter(fn {_dim, birth, death} -> birth < death end)
      |> Enum.sort()

    essential =
      raw_essential
      |> Enum.map(fn i ->
        s = Map.fetch!(idx_to_simplex, i)
        {s.dim, s.birth}
      end)
      |> Enum.sort()

    diagram =
      (pairs ++ Enum.map(essential, fn {d, b} -> {d, b, :infinity} end))
      |> Enum.sort()

    %{pairs: pairs, essential: essential, diagram: diagram}
  end
end
