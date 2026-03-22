import Config

config :nx, default_backend: EXLA.Backend
config :exla,
  preferred_clients: [:default],
  clients: [default: [platform: :host]]

if config_env() == :test, do: import_config "test.exs"
