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

  @typedoc "The result map returned by `compute/2`."
  @type result :: %{
          pairs: [pair()],
          essential: [essential_class()],
          diagram: [diagram_entry()]
        }

  @doc """
  Compute persistent homology for a point cloud.

  Options:
    - `max_dim` (integer, default 2): maximum simplex dimension in the filtration.
      To compute Hₖ you need simplices up to dimension k+1.
    - `threshold` (float, default: enclosing radius of the point cloud): ignore simplices
      born after this filtration value. The enclosing radius is the smallest value at which
      all points are connected, so the default produces the most topologically meaningful
      filtration. Pass `threshold: :infinity` to include all simplices regardless of scale.

  Returns a map:
    %{
      pairs:     [{dim, birth, death}],   # finite persistence pairs, sorted by dim then birth
      essential: [{dim, birth}],          # essential (infinite) classes
      diagram:   [{dim, birth, death | :infinity}]  # union of pairs + essential
    }
  """
  @spec compute(Nx.Tensor.t() | [[number()]], keyword()) :: result()
  def compute(points, opts \\ []) do
    opts = Keyword.validate!(opts, [:max_dim, :threshold])
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

    filtration =
      dist
      |> Filtration.build(max_dim)
      |> maybe_threshold(threshold)

    {boundary, _filtration} = BoundaryMatrix.build(filtration)

    %{pairs: raw_pairs, essential: raw_essential} =
      Reduction.reduce(boundary, length(filtration), filtration)

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
  @spec most_persistent(result(), pos_integer()) :: [
          {non_neg_integer(), float(), float(), float()}
        ]
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
