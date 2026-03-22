defmodule PhNx.Filtration do
  @moduledoc """
  Builds a Vietoris-Rips filtration from a distance matrix.

  A simplex σ = {v₀, v₁, …, vₖ} is born at time
    birth(σ) = max{ dist(vᵢ, vⱼ) | i ≠ j }
  i.e. the longest edge among all pairs of its vertices.

  The filtration is a list of simplices sorted by birth time, with ties broken by
  dimension (lower-dimensional simplices first) and then lexicographically by vertex list.
  Each simplex is represented as:

      %{vertices: [integer], dim: integer, birth: float, index: integer}

  where `index` is the 0-based position in the sorted filtration.

  Distance lookups use a flat tuple extracted from the Nx distance matrix, giving
  O(1) access per pair via `elem/2` instead of O(n) nested `Enum.at` traversals.
  """

  @doc """
  Build a Vietoris-Rips filtration up to `max_dim` from a precomputed distance matrix.

  `dist_matrix` is an n×n Nx tensor (or list-of-lists). `max_dim` controls the maximum
  simplex dimension included (0 = vertices only, 1 = edges, 2 = triangles, etc.).
  """
  def build(dist_matrix, max_dim \\ 2) do
    n = Nx.axis_size(dist_matrix, 0)

    # Extract the distance matrix into a flat tuple for O(1) pair lookups.
    # dist_flat[i*n + j] == dist(i, j). Nx.to_list handles EXLA device-to-host
    # transfer; the subsequent Elixir operations are allocation-free per simplex.
    dist_flat =
      dist_matrix
      |> Nx.to_list()
      |> Enum.concat()
      |> List.to_tuple()

    vertices = for i <- 0..(n - 1), do: [i]

    all_simplices =
      Enum.reduce(0..max_dim, vertices, fn dim, acc ->
        if dim == 0 do
          acc
        else
          prev_dim_simplices = Enum.filter(acc, fn s -> length(s) == dim end)

          new_simplices =
            for simplex <- prev_dim_simplices,
                v <- 0..(n - 1),
                v > List.last(simplex) do
              simplex ++ [v]
            end

          acc ++ new_simplices
        end
      end)

    all_simplices
    |> Enum.map(fn vertices ->
      # Compute birth = max pairwise distance via O(1) tuple lookup.
      # The `for ... reduce` accumulates the max without intermediate lists.
      birth =
        for i <- vertices, j <- vertices, i < j, reduce: 0.0 do
          acc -> max(acc, elem(dist_flat, i * n + j))
        end

      %{vertices: vertices, dim: length(vertices) - 1, birth: birth}
    end)
    |> Enum.sort_by(fn %{birth: b, dim: d, vertices: v} -> {b, d, v} end)
    |> Enum.with_index()
    |> Enum.map(fn {simplex, idx} -> Map.put(simplex, :index, idx) end)
  end

  @doc """
  Return all (k-1)-faces of a k-simplex (as sorted vertex lists).
  """
  def faces(%{vertices: vertices}) do
    vertices
    |> Enum.with_index()
    |> Enum.map(fn {_, i} ->
      List.delete_at(vertices, i)
    end)
  end

  @doc """
  Build a lookup map from vertex list to filtration index.
  """
  def index_map(filtration) do
    Map.new(filtration, fn %{vertices: v, index: i} -> {v, i} end)
  end
end
