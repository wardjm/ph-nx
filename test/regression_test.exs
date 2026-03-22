defmodule PhNx.RegressionTest do
  use ExUnit.Case, async: false

  # Known-correct values from ripser --format point-cloud --dim 1 test/fixtures/o3_50.txt
  @fixture "test/fixtures/o3_50.txt"

  defp load_fixture do
    File.read!(@fixture)
    |> String.split("\n", trim: true)
    |> Enum.map(fn line -> line |> String.split("\t") |> Enum.map(&String.to_float/1) end)
    |> Nx.tensor(type: :f64)
  end

  defp result, do: PhNx.compute(load_fixture(), max_dim: 2)

  test "enclosing radius threshold: simplex count is reduced vs unthresholded filtration" do
    pts = load_fixture()
    dist = PhNx.Distance.euclidean(pts)
    radius = PhNx.Distance.enclosing_radius(dist)
    filt = PhNx.Filtration.build(dist, 2)
    thresholded = Enum.filter(filt, fn %{birth: b} -> b <= radius end)
    # Threshold should cut at least some simplices (unthresholded is 20,875)
    assert length(thresholded) < length(filt)
    # Threshold matches ripser's reported enclosing radius of ~2.84927
    assert_in_delta radius, 2.84927, 1.0e-4
  end

  test "H0: 49 finite pairs (one per merge)" do
    h0_pairs = result().pairs |> Enum.filter(fn {d, _, _} -> d == 0 end)
    assert length(h0_pairs) == 49
  end

  test "H1: 51 finite pairs" do
    h1_pairs = result().pairs |> Enum.filter(fn {d, _, _} -> d == 1 end)
    assert length(h1_pairs) == 51
  end

  # Ripser ground truth: ripser --format point-cloud --dim 1 test/fixtures/o3_50.txt
  @ripser_h0_deaths [
    0.203848,
    0.333296,
    0.406143,
    0.49954,
    0.626872,
    0.704149,
    0.769835,
    0.77341,
    0.853078,
    0.866493,
    0.868387,
    0.945082,
    0.961139,
    0.975577,
    0.99351,
    1.01689,
    1.05871,
    1.06455,
    1.08609,
    1.11355,
    1.12017,
    1.14373,
    1.16802,
    1.1747,
    1.19196,
    1.2211,
    1.35095,
    1.36323,
    1.36514,
    1.36798,
    1.46941,
    1.48269,
    1.48402,
    1.49106,
    1.49738,
    1.55301,
    1.57482,
    1.59159,
    1.64608,
    1.68872,
    1.69053,
    1.69744,
    1.71336,
    1.79167,
    1.80458,
    1.86095,
    1.98702,
    2.0,
    2.00008
  ]

  @ripser_h1_pairs [
    {1.82781, 1.95276},
    {1.86636, 2.10998},
    {1.86776, 2.09713},
    {1.89679, 2.10545},
    {1.90625, 1.99817},
    {1.92644, 2.09271},
    {1.98327, 1.99817},
    {1.98919, 2.09024},
    {2.00001, 2.03603},
    {2.00001, 2.11115},
    {2.00002, 2.08364},
    {2.0001, 2.05943},
    {2.00018, 2.03079},
    {2.00021, 2.02743},
    {2.00025, 2.00101},
    {2.00027, 2.03578},
    {2.00072, 2.09732},
    {2.00098, 2.01669},
    {2.00104, 2.00846},
    {2.00118, 2.03603},
    {2.0017, 2.01584},
    {2.00175, 2.06196},
    {2.00182, 2.00467},
    {2.00192, 2.08925},
    {2.00196, 2.02202},
    {2.00209, 2.01086},
    {2.00227, 2.06196},
    {2.00238, 2.00427},
    {2.0027, 2.05089},
    {2.00293, 2.04112},
    {2.00461, 2.00749},
    {2.00561, 2.01293},
    {2.00568, 2.01293},
    {2.0062, 2.03854},
    {2.00658, 2.03213},
    {2.00718, 2.03767},
    {2.00941, 2.01248},
    {2.01035, 2.04122},
    {2.01413, 2.05693},
    {2.01577, 2.08529},
    {2.01954, 2.05511},
    {2.02115, 2.06943},
    {2.02397, 2.06137},
    {2.02731, 2.057},
    {2.02742, 2.03372},
    {2.02933, 2.06943},
    {2.03125, 2.03169},
    {2.03312, 2.07797},
    {2.04699, 2.09024},
    {2.09268, 2.11803},
    {2.14178, 2.16571}
  ]

  test "H1: birth and death values match ripser within 1e-4" do
    our_h1 =
      result().pairs
      |> Enum.filter(fn {d, _, _} -> d == 1 end)
      |> Enum.map(fn {_, b, death} -> {b, death} end)
      |> Enum.sort()

    Enum.zip(Enum.sort(@ripser_h1_pairs), our_h1)
    |> Enum.each(fn {{eb, ed}, {ab, ad}} ->
      assert_in_delta ab, eb, 1.0e-4, "H1 birth mismatch: expected #{eb}, got #{ab}"
      assert_in_delta ad, ed, 1.0e-4, "H1 death mismatch: expected #{ed}, got #{ad}"
    end)
  end

  test "H0: death values match ripser within 1e-4" do
    our_deaths =
      result().pairs
      |> Enum.filter(fn {d, _, _} -> d == 0 end)
      |> Enum.map(fn {_, _, death} -> death end)
      |> Enum.sort()

    Enum.zip(@ripser_h0_deaths, our_deaths)
    |> Enum.each(fn {expected, actual} ->
      assert_in_delta actual,
                      expected,
                      1.0e-4,
                      "H0 death mismatch: expected #{expected}, got #{actual}"
    end)
  end
end
