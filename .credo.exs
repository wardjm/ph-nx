# Credo configuration for ph_nx.
#
# This starts from Credo's default check set and only records deviations, so
# checks added by future Credo versions are picked up automatically rather than
# being pinned to whatever was current when the config was generated.
#
# Every entry under `checks:` is a deliberate deviation with a stated reason.
# Anything not listed runs at its default configuration under
# `mix credo --strict`.
%{
  configs: [
    %{
      name: "default",
      files: %{
        included: ["lib/", "test/", "bench/", "mix.exs"],
        excluded: [~r"/_build/", ~r"/deps/"]
      },
      plugins: [],
      requires: [],
      strict: true,
      parse_timeout: 5000,
      color: true,
      checks: [
        # This library is numeric kernel code: the boundary-matrix and
        # filtration builders are written as `Enum.reduce/3` pipelines whose
        # accumulator functions Credo counts as nesting levels. Depth 3 here
        # is one reducer with a `case`/`if` inside — the idiomatic shape, not
        # a refactoring signal. Depth 4 and beyond still fails.
        {Credo.Check.Refactor.Nesting, max_nesting: 3},

        # `Persistence.compute/2` and `Topology.compute/2` are the public
        # entry points and carry argument validation for the whole library: a
        # run of independent `raise ArgumentError` guards, each of which adds
        # a branch. The threshold is raised just past those two so genuinely
        # tangled logic elsewhere still trips the check.
        {Credo.Check.Refactor.CyclomaticComplexity, max_complexity: 12}
      ]
    }
  ]
}
