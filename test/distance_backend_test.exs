defmodule PhNx.DistanceBackendTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog

  alias PhNx.Distance

  @pts Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

  describe "euclidean/2 backend: :cpu" do
    test "explicit :cpu returns same matrix as default" do
      default = Distance.euclidean(@pts)
      explicit = Distance.euclidean(@pts, backend: :cpu)
      assert Nx.to_list(explicit) == Nx.to_list(default)
    end
  end

  describe "euclidean/2 backend: :gpu" do
    test "returns numerically correct distances (falls back to CPU when GPU unavailable)" do
      result = Distance.euclidean(@pts, backend: :gpu)
      assert Nx.shape(result) == {3, 3}
      mat = Nx.to_list(result)
      assert_in_delta Enum.at(Enum.at(mat, 0), 1), 1.0, 1.0e-9
      assert_in_delta Enum.at(Enum.at(mat, 1), 2), :math.sqrt(2), 1.0e-9
    end
  end

  describe "GPU fallback logging" do
    test "logs a warning when GPU backend is unavailable" do
      assert capture_log(fn -> Distance.euclidean(@pts, backend: :gpu) end) =~
               "GPU backend unavailable, falling back to CPU"
    end
  end
end
