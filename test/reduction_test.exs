defmodule PhNx.ReductionTest do
  use ExUnit.Case, async: true

  alias PhNx.{BoundaryMatrix, Reduction}

  @points Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

  describe "PhNx.Reduction.reduce/1 (BoundaryMatrix)" do
    test "reduces a boundary matrix correctly" do
      # Start with a matrix that's not reduced (simulated)
      # We use build_from_filtration without apparent pairs seeding to ensure we have work to do
      filtration = PhNx.Filtration.build(@points, 1)
      bm = BoundaryMatrix.build_from_filtration(filtration, seed_apparent: false)

      reduced_bm = Reduction.reduce(bm)

      assert reduced_bm.reduced
      assert reduced_bm.pairs != []
    end
  end

  describe "PhNx.Reduction.reduce/1 (Filtration)" do
    test "reduces a filtration directly" do
      filtration = PhNx.Filtration.build(@points, 1)
      reduced_bm = Reduction.reduce(filtration)

      assert reduced_bm.reduced
      assert reduced_bm.pairs != []
    end

    test "reduces a filtration with options" do
      filtration = PhNx.Filtration.build(@points, 1)
      reduced_bm = PhNx.reduction(filtration, seed_apparent: false)

      assert reduced_bm.reduced
      assert reduced_bm.pairs != []
    end
  end

  describe "regression: reduction with pre-seeded apparent pairs" do
    test "respects apparent pairs and completes reduction" do
      filtration = PhNx.Filtration.build(@points, 1)
      # Use default build which seeds apparent pairs
      bm = BoundaryMatrix.build_from_filtration(filtration)

      reduced_bm = Reduction.reduce(bm)

      assert reduced_bm.reduced
      # Since we seeded, we should at least have the pairs found during seeding
      assert reduced_bm.pairs != []
    end
  end
end
