import Config

config :nx, default_backend: EXLA.Backend

config :exla,
  preferred_clients: [:default],
  clients: [default: [platform: :host]]
