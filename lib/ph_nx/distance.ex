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
  Extract the upper-triangular entries of a distance matrix as `{i, j, dist}` triples,
  sorted by distance ascending.
  """
  def sorted_edges(dist_matrix) do
    n = Nx.axis_size(dist_matrix, 0)
    mat = Nx.to_list(dist_matrix)

    for i <- 0..(n - 2),
        j <- (i + 1)..(n - 1) do
      d = mat |> Enum.at(i) |> Enum.at(j)
      {i, j, d}
    end
    |> Enum.sort_by(fn {_, _, d} -> d end)
  end
end
