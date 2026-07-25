defmodule PhNx.DistanceBackendConfigTest do
  # Not async: these tests set :ph_nx, :distance_backend, which is global state
  # that any other test calling Distance.euclidean/1 would observe (issue #126).
  use ExUnit.Case, async: false

  alias PhNx.Distance

  @pts Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

  setup do
    on_exit(fn -> Application.delete_env(:ph_nx, :distance_backend) end)
  end

  describe "app config :distance_backend" do
    test ":gpu config produces correct distances" do
      Application.put_env(:ph_nx, :distance_backend, :gpu)

      result = Distance.euclidean(@pts)
      assert Nx.shape(result) == {3, 3}
      mat = Nx.to_list(result)
      assert_in_delta Enum.at(Enum.at(mat, 0), 1), 1.0, 1.0e-9
    end
  end
end
