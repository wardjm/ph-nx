defmodule PhNx.Distance do
  @moduledoc """
  Pairwise distance matrix computation using Nx.

  Supports Euclidean distance for point clouds given as n×d tensors (n points, d dimensions).
  """

  import Nx.Defn

  @doc """
  Compute the n×n pairwise Euclidean distance matrix for an n×d point cloud.

  ## Example

      iex> points = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      iex> PhNx.Distance.euclidean(points)
      #Nx.Tensor<...>
  """
  def euclidean(points) do
    points = Nx.as_type(points, :f64)
    euclidean_n(points)
  end

  defnp euclidean_n(points) do
    # points: {n, d}
    # Expand to {n, 1, d} and {1, n, d} for broadcasting
    n = Nx.axis_size(points, 0)
    a = Nx.reshape(points, {n, 1, Nx.axis_size(points, 1)})
    b = Nx.reshape(points, {1, n, Nx.axis_size(points, 1)})
    diff = a - b
    # Sum of squares along last axis, then sqrt
    Nx.sqrt(Nx.sum(diff * diff, axes: [-1]))
  end

  @doc """
  Compute the enclosing radius of a distance matrix: the smallest ε at which some point
  is within distance ε of all other points.

  Formula: min over all rows of (max over all columns).

  Above this scale no new topology can appear, so the Vietoris-Rips filtration can be
  safely truncated here.
  """
  def enclosing_radius(dist_matrix) do
    dist_matrix
    |> Nx.reduce_max(axes: [1])
    |> Nx.reduce_min()
    |> Nx.to_number()
  end

  @doc """
  Extract the upper-triangular entries of a distance matrix as `{i, j, dist}` triples,
  sorted by distance ascending.

  Useful as a user-facing utility for inspecting the edge order of a Vietoris-Rips
  filtration without building the full filtration.

  Uses a flat-tuple representation for O(1) per-pair access, giving O(n²) total
  rather than the O(n³) cost of nested list traversals.
  """
  def sorted_edges(dist_matrix) do
    n = Nx.axis_size(dist_matrix, 0)
    dist_flat = dist_matrix |> Nx.to_list() |> Enum.concat() |> List.to_tuple()

    for i <- 0..(n - 2),
        j <- (i + 1)..(n - 1) do
      {i, j, elem(dist_flat, i * n + j)}
    end
    |> Enum.sort_by(fn {_, _, d} -> d end)
  end
end
