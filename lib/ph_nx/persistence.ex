defmodule PhNx.Persistence do
  @moduledoc """
  High-level persistent homology computation.

  Given a point cloud (as an Nx tensor or list of coordinate lists), computes:
    - Persistence pairs: {dimension, birth, death} for each finite bar
    - Essential classes:  {dimension, birth} for each infinite bar (death = ∞)

  Uses the Vietoris-Rips filtration and the standard persistence algorithm over F₂.

  ## Example

      points = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
      result = PhNx.Persistence.compute(points)
      PhNx.Persistence.print_barcode(result)
  """

  alias PhNx.{Distance, Filtration, BoundaryMatrix, Reduction}

  @typedoc "A finite persistence pair: {dimension, birth, death}."
  @type pair :: {non_neg_integer(), float(), float()}

  @typedoc "An essential (infinite) homology class: {dimension, birth}."
  @type essential_class :: {non_neg_integer(), float()}

  @typedoc "A diagram entry: finite pair or essential class extended to :infinity."
  @type diagram_entry :: {non_neg_integer(), float(), float() | :infinity}

  @typedoc "A most-persistent feature: {dimension, birth, death, persistence} where persistence = death - birth."
  @type persistent_feature :: {non_neg_integer(), float(), float(), float()}

  @typedoc "The result map returned by `compute/2`."
  @type result :: %{
          pairs: [pair()],
          essential: [essential_class()],
          diagram: [diagram_entry()]
        }

  @typedoc "A function that builds a BoundaryMatrix from a filtration and options."
  @type boundary_builder :: ([Filtration.simplex()], keyword() -> BoundaryMatrix.t())

  @doc """
  Compute persistent homology for a point cloud.

  Options:
    - `max_dim` (integer, default 2): maximum simplex dimension in the filtration.
      To compute Hₖ you need simplices up to dimension k+1.
    - `threshold` (float, default: enclosing radius of the point cloud): ignore simplices
      born after this filtration value. The enclosing radius is the smallest value at which
      all points are connected, so the default produces the most topologically meaningful
      filtration. Pass `threshold: :infinity` to include all simplices regardless of scale.
    - `boundary_builder` (`(filtration, opts -> BoundaryMatrix.t())`, default:
      `&BoundaryMatrix.build_from_filtration/2`): function used to construct the boundary
      matrix from the filtered simplex list. Override to inject alternative implementations
      (e.g. GPU-accelerated, pre-seeded, or test doubles).

  Returns a map:
    %{
      pairs:     [{dim, birth, death}],   # finite persistence pairs, sorted by dim then birth
      essential: [{dim, birth}],          # essential (infinite) classes
      diagram:   [{dim, birth, death | :infinity}]  # union of pairs + essential
    }
  """
  @spec compute(Nx.Tensor.t() | [[number()]], keyword()) :: result()
  def compute(points, opts \\ []) do
    opts = Keyword.validate!(opts, [:max_dim, :threshold, :boundary_builder])
    max_dim = Keyword.get(opts, :max_dim, 2)

    if not is_integer(max_dim) or max_dim < 0 do
      raise ArgumentError, "max_dim must be a non-negative integer, got: #{inspect(max_dim)}"
    end

    if points == [] or (is_list(points) and length(points) == 0) do
      raise ArgumentError, "point cloud must be non-empty"
    end

    points = if is_list(points), do: Nx.tensor(points, type: :f64), else: points

    if Nx.axis_size(points, 0) == 0 do
      raise ArgumentError, "point cloud must be non-empty"
    end

    dist = Distance.euclidean(points)

    threshold =
      Keyword.get_lazy(opts, :threshold, fn -> Distance.enclosing_radius(dist) end)

    unless threshold == :infinity or (is_number(threshold) and threshold >= 0) do
      raise ArgumentError,
            "threshold must be :infinity or a non-negative number, got: #{inspect(threshold)}"
    end

    filtration =
      dist
      |> Filtration.build(max_dim)
      |> maybe_threshold(threshold)

    builder = Keyword.get(opts, :boundary_builder, &BoundaryMatrix.build_from_filtration/2)

    unless is_function(builder, 2) do
      raise ArgumentError, "boundary_builder must be a 2-arity function, got: #{inspect(builder)}"
    end

    builder_opts = Keyword.delete(opts, :boundary_builder)
    reduced = builder.(filtration, builder_opts) |> BoundaryMatrix.reduce()
    raw_pairs = BoundaryMatrix.pairs(reduced)
    raw_essential = BoundaryMatrix.essential(reduced)

    # Map filtration indices back to (dim, birth) info
    idx_to_simplex = Map.new(filtration, fn s -> {s.index, s} end)

    pairs =
      raw_pairs
      |> Enum.map(fn {i, j} ->
        s_i = Map.fetch!(idx_to_simplex, i)
        s_j = Map.fetch!(idx_to_simplex, j)
        # The homology class lives in dimension = dim of the creator (lower dim simplex)
        dim = s_i.dim
        {dim, s_i.birth, s_j.birth}
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

  defp maybe_threshold(filtration, :infinity), do: filtration

  defp maybe_threshold(filtration, threshold) do
    Enum.filter(filtration, fn %{birth: b} -> b <= threshold end)
  end

  @doc """
  Compute persistent homology for a point cloud, using lazy filtration streaming.

  This function accepts an `Enumerable.t()` of points — a plain list, a `Stream`, or any
  other lazy collection — so callers that already have points behind an enumerable
  (file-backed iterators, `IO.stream/2`, resource streams) can pass it directly instead
  of collecting it themselves first.

  ## Performance

  This is a convenience over `compute/2`, not an optimisation. It is currently **slower
  than `compute/2` and uses comparable memory**; prefer `compute/2` when you already hold
  a list or tensor.

  The enumerable is fully materialised before computation: the distance matrix requires
  all points, and although the Vietoris-Rips filtration is generated via
  `FiltrationBuilder.stream/2`, it is then sorted globally, which realises it in full.
  Measured on 55 points (`max_dim: 2, threshold: :infinity`), peak memory was 45.4 MB
  versus 48.1 MB for `compute/2` — within noise — while wall-clock was 1.4-2x slower,
  because `FiltrationBuilder.stream/2` reads birth times out of the distance tensor
  per vertex pair rather than from a flat tuple.

  Results are numerically identical to `compute/2` for the same options.

  ## Options

  Same as `compute/2`:
    - `:max_dim` (default `2`)
    - `:threshold` (default: enclosing radius)
    - `:boundary_builder` (default: `&BoundaryMatrix.build_from_filtration/2`)
    - `:coeff` — coefficient ring; `{:zp, p}` for ℤₚ arithmetic (default: ℤ₂)
    - `:on_progress` — 1-arity callback invoked once per column during reduction

  Note that `:backend` is accepted by `PhNx.Topology.compute/2` but not here.

  ## Examples

      iex> points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
      iex> result = PhNx.Persistence.compute_stream(Stream.map(points, & &1), max_dim: 1)
      iex> result == PhNx.compute(points, max_dim: 1)
      true
  """
  @spec compute_stream(Enumerable.t(), keyword()) :: result()
  def compute_stream(points_stream, opts \\ []) do
    opts =
      Keyword.validate!(opts, [:max_dim, :threshold, :boundary_builder, :coeff, :on_progress])

    max_dim = Keyword.get(opts, :max_dim, 2)

    if not is_integer(max_dim) or max_dim < 0 do
      raise ArgumentError, "max_dim must be a non-negative integer, got: #{inspect(max_dim)}"
    end

    # The distance matrix needs every point, so the enumerable is realised in full here.
    points_list = Enum.to_list(points_stream)

    if points_list == [] do
      raise ArgumentError, "point cloud must be non-empty"
    end

    points = Nx.tensor(points_list, type: :f64)

    dist = Distance.euclidean(points)

    threshold =
      Keyword.get_lazy(opts, :threshold, fn -> Distance.enclosing_radius(dist) end)

    unless threshold == :infinity or (is_number(threshold) and threshold >= 0) do
      raise ArgumentError,
            "threshold must be :infinity or a non-negative number, got: #{inspect(threshold)}"
    end

    builder = Keyword.get(opts, :boundary_builder, &BoundaryMatrix.build_from_filtration/2)

    unless is_function(builder, 2) do
      raise ArgumentError, "boundary_builder must be a 2-arity function, got: #{inspect(builder)}"
    end

    builder_opts = Keyword.delete(opts, :boundary_builder)

    # Use FiltrationBuilder.stream/2 for lazy filtration generation, then sort
    # globally (like build/2 does) and add :index fields so the result is
    # compatible with BoundaryMatrix.build_from_filtration/2
    filtration =
      points
      |> PhNx.FiltrationBuilder.stream(max_dim: max_dim, threshold: threshold)
      |> Enum.sort_by(fn %{birth: b, dim: d, vertices: v} -> {b, d, v} end)
      |> Enum.with_index()
      |> Enum.map(fn {simplex, idx} -> Map.put(simplex, :index, idx) end)

    reduced = builder.(filtration, builder_opts) |> Reduction.reduce(builder_opts)
    raw_pairs = BoundaryMatrix.pairs(reduced)
    raw_essential = BoundaryMatrix.essential(reduced)

    # Map filtration indices back to (dim, birth) info
    idx_to_simplex = Map.new(filtration, fn s -> {s.index, s} end)

    pairs =
      raw_pairs
      |> Enum.map(fn {i, j} ->
        s_i = Map.fetch!(idx_to_simplex, i)
        s_j = Map.fetch!(idx_to_simplex, j)
        # The homology class lives in dimension = dim of the creator (lower dim simplex)
        dim = s_i.dim
        {dim, s_i.birth, s_j.birth}
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

  @doc """
  Print a human-readable barcode summary.
  """
  @spec print_barcode(result()) :: :ok
  def print_barcode(%{diagram: diagram}) do
    IO.puts("\nPersistence Barcode")
    IO.puts(String.duplicate("─", 50))

    diagram
    |> Enum.group_by(fn {d, _, _} -> d end)
    |> Enum.sort_by(fn {d, _} -> d end)
    |> Enum.each(fn {dim, bars} ->
      IO.puts("\nH#{dim} (#{length(bars)} bar#{if length(bars) == 1, do: "", else: "s"}):")

      bars
      |> Enum.sort_by(fn {_, b, _} -> b end)
      |> Enum.each(fn
        {_, birth, :infinity} ->
          IO.puts("  [#{fmt(birth)}, ∞)")

        {_, birth, death} ->
          persistence = death - birth
          IO.puts("  [#{fmt(birth)}, #{fmt(death)})  persistence: #{fmt(persistence)}")
      end)
    end)

    IO.puts("")
  end

  @doc """
  Return persistence pairs sorted by persistence (death - birth) descending.
  Useful for identifying the most significant topological features.
  """
  @spec most_persistent(result(), pos_integer()) :: [persistent_feature()]
  def most_persistent(%{pairs: pairs}, n \\ 10) do
    pairs
    |> Enum.map(fn {dim, birth, death} -> {dim, birth, death, death - birth} end)
    |> Enum.sort_by(fn {_, _, _, p} -> -p end)
    |> Enum.take(n)
  end

  @doc """
  Compute Betti numbers from the essential classes.
  Returns a map `%{dimension => count}`.
  """
  @spec betti_numbers(result()) :: %{optional(non_neg_integer()) => non_neg_integer()}
  def betti_numbers(%{essential: essential}) do
    essential
    |> Enum.group_by(fn {d, _} -> d end)
    |> Map.new(fn {d, list} -> {d, length(list)} end)
  end

  defp fmt(x) when is_float(x), do: :io_lib.format("~.4f", [x]) |> IO.iodata_to_binary()
  defp fmt(x) when is_integer(x), do: Integer.to_string(x)
end
