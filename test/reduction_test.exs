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

  describe "PhNx.Reduction.reduce/2 with coeff: :z2" do
    test "explicit :z2 gives same pairs as default" do
      filtration = PhNx.Filtration.build(@points, 1)

      default_pairs = Reduction.reduce(filtration) |> BoundaryMatrix.pairs()
      z2_pairs = Reduction.reduce(filtration, coeff: :z2) |> BoundaryMatrix.pairs()

      assert z2_pairs == default_pairs
    end

    test "accepts BoundaryMatrix input with coeff: :z2" do
      filtration = PhNx.Filtration.build(@points, 1)
      bm = BoundaryMatrix.build_from_filtration(filtration, seed_apparent: false)

      reduced = Reduction.reduce(bm, coeff: :z2)

      assert reduced.reduced
      assert reduced.pairs != []
    end
  end

  describe "PhNx.Reduction.reduce/2 with coeff: {:zp, p}" do
    test "reduces a filtration over Z3 and returns a reduced matrix" do
      filtration = PhNx.Filtration.build(@points, 1)

      reduced = Reduction.reduce(filtration, coeff: {:zp, 3})

      assert reduced.reduced
    end

    test "Z3 produces the same persistence pairs as Z2 for a simple triangle" do
      filtration = PhNx.Filtration.build(@points, 1)

      z2_pairs = Reduction.reduce(filtration, coeff: :z2) |> BoundaryMatrix.pairs()
      z3_pairs = Reduction.reduce(filtration, coeff: {:zp, 3}) |> BoundaryMatrix.pairs()

      assert MapSet.new(z2_pairs) == MapSet.new(z3_pairs)
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
