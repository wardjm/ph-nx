defmodule PhNx.FiltrationBuilderTest do
  use ExUnit.Case

  alias PhNx.{FiltrationBuilder, Filtration}

  @tri [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0]
  ]

  test "build full filtration without threshold" do
    filtration = FiltrationBuilder.build(@tri, max_dim: 2)
    assert length(filtration) == 7
    vertex_0 = Enum.find(filtration, fn s -> s.dim == 0 and s.vertices == [0] end)
    assert vertex_0.birth == 0.0
    e01 = Enum.find(filtration, fn s -> s.vertices == [0, 1] end)
    assert e01.birth == 1.0
    e02 = Enum.find(filtration, fn s -> s.vertices == [0, 2] end)
    assert e02.birth == 1.0
    e12 = Enum.find(filtration, fn s -> s.vertices == [1, 2] end)
    assert e12.birth > 1.4 and e12.birth < 1.5
  end

  test "threshold filters simplices beyond birth value" do
    filtration = FiltrationBuilder.build(@tri, max_dim: 2, threshold: 1.0)
    assert Enum.all?(filtration, fn s -> s.birth <= 1.0 end)
    assert length(filtration) == 5
    assert Enum.none?(filtration, fn s -> s.vertices == [1, 2] end)
    assert Enum.none?(filtration, fn s -> s.vertices == [0, 1, 2] end)
  end

  test "threshold :infinity returns full filtration" do
    full = FiltrationBuilder.build(@tri, max_dim: 2)
    full_inf = FiltrationBuilder.build(@tri, max_dim: 2, threshold: :infinity)
    assert full == full_inf
  end

  test "max_dim 1 returns only vertices and edges" do
    filt = FiltrationBuilder.build(@tri, max_dim: 1)
    assert length(filt) == 6
    assert Enum.all?(filt, fn s -> s.dim <= 1 end)
  end
end
