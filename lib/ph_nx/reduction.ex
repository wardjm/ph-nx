defmodule PhNx.Reduction do
  @moduledoc """
  Boundary matrix reduction for persistent homology.

  Supports ℤ₂ (default) and arbitrary ℤₚ coefficient rings. Pass `coeff: {:zp, p}` to
  `reduce/2` to use ℤₚ arithmetic with signed boundary operators.
  """

  alias PhNx.BoundaryMatrix

  @doc """
  Reduces a boundary matrix or a filtration to find persistence pairs.

  If a `BoundaryMatrix` is provided, it performs the standard column-reduction algorithm.
  If a filtration (list of simplices) is provided, it first builds the boundary matrix
  from the filtration and then performs the reduction.

  Returns the reduced boundary matrix.

  ## Options

  - `:on_progress` — a 1-arity callback invoked once per column during reduction.
    Receives `%{current: non_neg_integer(), total: pos_integer()}`.
  - `:coeff` — coefficient ring; `{:zp, p}` for ℤₚ arithmetic (default: ℤ₂).
  - `:seed_apparent` — when `true`, seeds apparent pairs before reduction (default: `false`).
  """
  @spec reduce(BoundaryMatrix.t() | list(PhNx.Filtration.simplex()), keyword()) ::
          BoundaryMatrix.t()
  def reduce(input, opts \\ [])

  def reduce(%BoundaryMatrix{} = bm, opts) do
    BoundaryMatrix.reduce(bm, opts)
  end

  def reduce(filtration, opts) when is_list(filtration) do
    {progress_opts, build_opts} = Keyword.split(opts, [:on_progress])

    filtration
    |> BoundaryMatrix.build_from_filtration(build_opts)
    |> BoundaryMatrix.reduce(progress_opts)
  end

  def reduce(other, _opts) do
    raise ArgumentError,
          "PhNx.Reduction.reduce/2 expects a BoundaryMatrix or a list of simplices, but got #{inspect(other)}"
  end
end
