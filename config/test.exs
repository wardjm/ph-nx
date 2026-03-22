import Config

# Use the pure-Elixir backend in tests so the suite does not require
# the EXLA NIF (XLA compiled binaries) to be present.
config :nx, default_backend: Nx.BinaryBackend
