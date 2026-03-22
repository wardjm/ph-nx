import Config

# Use EXLA for accelerated computation during development.
# Requires XLA binaries — see the EXLA README if this fails to start.
config :nx, default_backend: EXLA.Backend

config :exla,
  preferred_clients: [:default],
  clients: [default: [platform: :host]]
