defmodule PhNx.Distance do
  @moduledoc """
  Pairwise distance matrix computation using Nx.

  Supports Euclidean distance for point clouds given as n×d tensors (n points, d dimensions).

  ## GPU acceleration

  Pass `backend: :gpu` to `euclidean/2` (or set `config :ph_nx, distance_backend: :gpu`) to
  JIT-compile via EXLA. The GPU client defaults to `:cuda`; override with
  `config :ph_nx, gpu_client: :rocm` (or another EXLA client name) for non-CUDA hardware.
  If the GPU is unavailable, the function logs a warning and falls back to CPU.
  """

  import Nx.Defn

  @doc """
  Compute the n×n pairwise Euclidean distance matrix for an n×d point cloud.

  ## Options

    - `:backend` (`:cpu` or `:gpu`, default `:cpu`) — computation backend. `:gpu` uses
      EXLA JIT compilation; falls back to CPU with a warning if the GPU is unavailable.
      The default backend can be overridden globally via `config :ph_nx, distance_backend: :gpu`.

  ## Example

      iex> points = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      iex> PhNx.Distance.euclidean(points)
      #Nx.Tensor<...>
  """
  @spec euclidean(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def euclidean(points, opts \\ []) do
    points = Nx.as_type(points, :f64)

    case resolve_backend(opts) do
      :gpu -> euclidean_gpu(points)
      _ -> euclidean_n(points)
    end
  end

  defp resolve_backend(opts) do
    Keyword.get(opts, :backend, Application.get_env(:ph_nx, :distance_backend, :cpu))
  end

  defp euclidean_gpu(points) do
    client = Application.get_env(:ph_nx, :gpu_client, :cuda)

    try do
      Nx.Defn.jit_apply(&euclidean_n/1, [points], compiler: EXLA, client: client)
    rescue
      e in RuntimeError ->
        gpu_fallback_warning(client, Exception.message(e))
        euclidean_n(points)
    catch
      :exit, reason ->
        gpu_fallback_warning(client, inspect(reason))
        euclidean_n(points)
    end
  end

  defp gpu_fallback_warning(client, detail) do
    require Logger

    Logger.warning(
      "PhNx.Distance: GPU backend unavailable, falling back to CPU (client: #{client}): #{detail}"
    )
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
  @spec enclosing_radius(Nx.Tensor.t()) :: float()
  def enclosing_radius(dist_matrix) do
    dist_matrix
    |> Nx.reduce_max(axes: [1])
    |> Nx.reduce_min()
    |> Nx.to_number()
  end

  @doc """
  Flatten a distance matrix Nx tensor into a tuple for O(1) pair lookups.

  `dist_flat[i*n + j]` gives `dist(i, j)` via `elem/2`.
  Handles EXLA device-to-host transfer via `Nx.to_list/1`.
  """
  @spec flat_tuple(Nx.Tensor.t()) :: tuple()
  def flat_tuple(dist_matrix) do
    dist_matrix |> Nx.to_list() |> Enum.concat() |> List.to_tuple()
  end

  @doc """
  Extract the upper-triangular entries of a distance matrix as `{i, j, dist}` triples,
  sorted by distance ascending.

  Useful as a user-facing utility for inspecting the edge order of a Vietoris-Rips
  filtration without building the full filtration.

  Uses a flat-tuple representation for O(1) per-pair access, giving O(n²) total
  rather than the O(n³) cost of nested list traversals.
  """
  @spec sorted_edges(Nx.Tensor.t()) :: [{non_neg_integer(), non_neg_integer(), float()}]
  def sorted_edges(dist_matrix) do
    n = Nx.axis_size(dist_matrix, 0)
    dist_flat = flat_tuple(dist_matrix)

    for i <- 0..(n - 2),
        j <- (i + 1)..(n - 1) do
      {i, j, elem(dist_flat, i * n + j)}
    end
    |> Enum.sort_by(fn {_, _, d} -> d end)
  end
end
