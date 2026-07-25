defmodule PhNx.MixProject do
  use Mix.Project

  def project do
    [
      app: :ph_nx,
      version: "0.1.0",
      elixir: "~> 1.18",
      description: "Persistent homology computation for Elixir using Nx",
      source_url: "https://github.com/wardjm/ph-nx",
      homepage_url: "https://github.com/wardjm/ph-nx",
      # app: nil so the escript starts no applications of its own. Starting
      # :exla from inside an archive raises a NIF load error before main/1 is
      # ever reached; the CLI picks its backend explicitly instead.
      escript: [main_module: PhNx.CLI, app: nil],
      docs: [
        main: "readme",
        extras: ["README.md", "docs/performance.md"]
      ],
      package: [
        licenses: ["MIT"],
        links: %{"GitHub" => "https://github.com/wardjm/ph-nx"},
        # Listed explicitly because Hex ships priv/ by default, and priv/plts
        # holds the (large, machine-specific) Dialyzer PLTs.
        files: ~w(lib mix.exs mix.lock README.md LICENSE docs)
      ],
      start_permanent: Mix.env() == :prod,
      # Keep the PLT inside the repo so CI can cache it as a single directory;
      # a cold build takes several minutes.
      dialyzer: [
        plt_local_path: "priv/plts",
        plt_core_path: "priv/plts",
        # :mix is needed for the Mix.Task behaviour behind `mix ph_nx`; it is
        # not a runtime dependency of the library so it is not pulled in
        # automatically.
        plt_add_apps: [:mix],
        flags: [:error_handling, :extra_return, :missing_return]
      ],
      deps: deps()
    ]
  end

  # Build the escript in :prod. An escript is a single archive, so the priv/
  # directory of :exla is never on disk and its NIF cannot be loaded; :prod is
  # the environment that does not configure EXLA as the default backend.
  def cli do
    [preferred_envs: ["escript.build": :prod]]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.9"},
      # EXLA is an optional accelerator (see the README): consumers add it
      # themselves. Keeping it out of :prod also keeps it out of the escript,
      # which cannot load its NIF from inside the archive.
      {:exla, "~> 0.11.0", only: [:dev, :test], runtime: Mix.env() != :test},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end
end
