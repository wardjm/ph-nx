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
      docs: [
        main: "readme",
        extras: ["README.md"]
      ],
      package: [
        licenses: ["MIT"],
        links: %{"GitHub" => "https://github.com/wardjm/ph-nx"}
      ],
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
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
      {:exla, "~> 0.11.0", runtime: Mix.env() != :test},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false}
    ]
  end
end
