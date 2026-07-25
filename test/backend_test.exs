defmodule PhNx.BackendTest do
  # install/1 sets the global default backend, so this suite cannot run
  # concurrently with suites that compute tensors.
  use ExUnit.Case, async: false

  alias PhNx.Backend

  defp available, do: fn _backend -> true end
  defp unavailable, do: fn _backend -> false end

  describe "select/2 when no accelerated backend is configured" do
    test "uses the binary backend without probing" do
      probe = fn _ -> flunk("an unaccelerated backend should not be probed") end

      assert {:ok, Nx.BinaryBackend} = Backend.select(nil, probe)
      assert {:ok, Nx.BinaryBackend} = Backend.select(Nx.BinaryBackend, probe)
      assert {:ok, Nx.BinaryBackend} = Backend.select({Nx.BinaryBackend, []}, probe)
    end
  end

  describe "select/2 when an accelerated backend is configured" do
    test "keeps the configured backend when it is usable" do
      assert {:ok, EXLA.Backend} = Backend.select(EXLA.Backend, available())

      assert {:ok, {EXLA.Backend, [client: :host]}} =
               Backend.select({EXLA.Backend, [client: :host]}, available())
    end

    test "probes the configured backend, not a hardcoded one" do
      probe = fn backend -> send(self(), {:probed, backend}) && true end

      Backend.select(Torchx.Backend, probe)
      assert_received {:probed, Torchx.Backend}
    end

    test "falls back to the binary backend when it is not usable" do
      assert {:fallback, Nx.BinaryBackend, reason} = Backend.select(EXLA.Backend, unavailable())
      assert reason =~ "EXLA.Backend"
      assert reason =~ "Nx.BinaryBackend"
    end

    test "names the configured backend in the fallback reason" do
      assert {:fallback, Nx.BinaryBackend, reason} =
               Backend.select({Torchx.Backend, [device: :cuda]}, unavailable())

      assert reason =~ "Torchx.Backend"
    end
  end

  describe "install/1" do
    setup do
      previous = Application.get_env(:nx, :default_backend)
      on_exit(fn -> Application.put_env(:nx, :default_backend, previous) end)
      :ok
    end

    test "installs the selected backend and returns the decision unchanged" do
      assert {:ok, Nx.BinaryBackend} = Backend.install({:ok, Nx.BinaryBackend})
      assert {Nx.BinaryBackend, []} = Application.get_env(:nx, :default_backend)

      assert {:fallback, Nx.BinaryBackend, "boom"} =
               Backend.install({:fallback, Nx.BinaryBackend, "boom"})

      assert {Nx.BinaryBackend, []} = Application.get_env(:nx, :default_backend)
    end
  end
end
