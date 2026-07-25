defmodule PhNx.Backend do
  @moduledoc """
  Chooses the `Nx` backend the CLI computes with.

  The library itself never selects a backend — that is the host application's
  job. The CLI is the host application for `mix ph_nx` and for the escript, so
  it has to make the choice, and it has to make it without crashing when an
  accelerated backend cannot be loaded.

  Two situations matter:

    * The escript. An escript is a single archive, so the `priv/` directory of
      a backend like `:exla` is never written to disk and `libexla.so` can
      never be `dlopen`ed. The escript is built with `MIX_ENV=prod`, where no
      accelerated backend is configured or even bundled, so it runs on
      `Nx.BinaryBackend`.

    * A development checkout on a machine without working XLA binaries. Here
      `config/dev.exs` asks for `EXLA.Backend` and starting `:exla` fails.
      Rather than let that surface as an Erlang crash report, fall back to
      `Nx.BinaryBackend` and say so on stderr.

  `select/2` decides; `install/1` applies the decision. They are separate so
  the decision can be tested without mutating global backend state.
  """

  @typedoc "A backend as `Nx` accepts it: a module, or a module with options."
  @type backend :: module() | {module(), keyword()}

  @typedoc """
  `{:ok, backend}` when the configured backend is usable, `{:fallback,
  backend, reason}` when it is not and `reason` explains the substitution.
  """
  @type decision :: {:ok, backend()} | {:fallback, backend(), String.t()}

  @doc """
  Decides which backend to use.

  `configured` is the backend the application environment asks for (`nil` when
  nothing is configured); `usable?` is the probe that reports whether a given
  backend can actually be loaded and started. Both are injectable for testing.
  """
  @spec select(backend() | nil, (backend() -> boolean())) :: decision()
  def select(configured \\ Application.get_env(:nx, :default_backend), usable? \\ &usable?/1) do
    cond do
      binary_backend?(configured) ->
        {:ok, Nx.BinaryBackend}

      usable?.(configured) ->
        {:ok, configured}

      true ->
        {:fallback, Nx.BinaryBackend,
         "#{inspect(module(configured))} is configured but could not be loaded; " <>
           "falling back to Nx.BinaryBackend (correct results, but slower). " <>
           "See the Backends section of the README."}
    end
  end

  @doc """
  Installs the backend from a `select/2` decision and returns the decision
  unchanged, so callers can report a fallback to the user.
  """
  @spec install(decision()) :: decision()
  def install(decision) do
    backend = elem(decision, 1)

    # Nx.global_default_backend/1 reads the current value first, which needs
    # :nx to be loaded. The escript starts no applications, so load it here.
    Application.ensure_loaded(:nx)
    Nx.global_default_backend(backend)
    decision
  end

  defp binary_backend?(nil), do: true
  defp binary_backend?(backend), do: module(backend) == Nx.BinaryBackend

  defp module({mod, _opts}), do: mod
  defp module(mod), do: mod

  # A backend module lives in the application named after its first segment:
  # EXLA.Backend -> :exla, Torchx.Backend -> :torchx. Starting it is the real
  # test — EXLA loads fine as a module and only fails when its NIF is missing,
  # which raises rather than returning an error tuple.
  defp usable?(configured) do
    mod = module(configured)

    Code.ensure_loaded?(mod) and
      match?({:ok, _}, Application.ensure_all_started(backend_app(mod)))
  rescue
    _ -> false
  catch
    :exit, _ -> false
  end

  defp backend_app(mod) do
    mod |> Module.split() |> hd() |> Macro.underscore() |> String.to_atom()
  end
end
