#!/usr/bin/env bash
# End-to-end tests for the built escript.
#
# Unlike test/cli_test.sh (which runs the mix task in-process), this builds the
# distribution artifact with `mix escript.build` and executes it. That is the
# only way to catch packaging failures — an escript cannot carry the EXLA NIF,
# so a runtime-started :exla makes every invocation fail (issue #130).
set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck source=test/support/sh_assert.sh
source "test/support/sh_assert.sh"

TMPDIR_CUSTOM=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR_CUSTOM"; }
trap cleanup EXIT

SQUARE="$TMPDIR_CUSTOM/square.txt"
cat > "$SQUARE" <<'EOF'
0.0,0.0
1.0,0.0
1.0,1.0
0.0,1.0
EOF

EMPTY="$TMPDIR_CUSTOM/empty.txt"
printf '# only comments\n\n' > "$EMPTY"

# ── build ───────────────────────────────────────────────────────────────────
mix escript.build >/dev/null
ESCRIPT="./ph_nx"
[ -x "$ESCRIPT" ] || { echo "FAIL: escript was not built"; exit 1; }

# ── packaging ───────────────────────────────────────────────────────────────
# The escript must not bundle a NIF-backed backend: its priv/ directory is
# never unpacked, so the NIF could not be loaded even if it were shipped.
if grep -qa "Elixir.EXLA" "$ESCRIPT"; then
  fail "escript bundles no EXLA beams"
else
  pass "escript bundles no EXLA beams"
fi

# ── startup ─────────────────────────────────────────────────────────────────
# These fail on a machine without XLA libraries if :exla is started by the
# escript, regardless of the arguments given.
assert_exit      "--help exits 0"                  0        "$ESCRIPT" --help
assert_stdout    "--help prints Usage"             "Usage"  "$ESCRIPT" --help
assert_no_stderr "--help writes nothing to stderr"          "$ESCRIPT" --help

# ── file mode ───────────────────────────────────────────────────────────────
assert_exit      "file mode exits 0"                0             "$ESCRIPT" "$SQUARE"
assert_stdout    "file mode prints point count"     "4 points"    "$ESCRIPT" "$SQUARE"
assert_stdout    "file mode prints H0 bars"         "H0 (4 bars)" "$ESCRIPT" "$SQUARE"
assert_stdout    "file mode prints H1 bar"          "H1 (1 bar)"  "$ESCRIPT" "$SQUARE"
assert_stdout    "file mode prints Betti numbers"   "β0 = 1"      "$ESCRIPT" "$SQUARE"
assert_no_stderr "file mode writes nothing to stderr"            "$ESCRIPT" "$SQUARE"

assert_exit      "--max-dim accepted"               0  "$ESCRIPT" "$SQUARE" --max-dim 1
assert_exit      "--threshold accepted"             0  "$ESCRIPT" "$SQUARE" --threshold 5.0

# ── stream mode ─────────────────────────────────────────────────────────────
stream_out=$("$ESCRIPT" --stream < "$SQUARE" 2>/dev/null)
file_out=$("$ESCRIPT" "$SQUARE" 2>/dev/null)

if echo "$stream_out" | grep -q "H1 (1 bar)"; then
  pass "stream mode computes the barcode"
else
  fail "stream mode computes the barcode (got: $stream_out)"
fi

if [ "$stream_out" = "$file_out" ]; then
  pass "stream and file modes agree"
else
  fail "stream and file modes agree"
fi

# ── error paths ─────────────────────────────────────────────────────────────
assert_exit   "no args exits 1"              1  "$ESCRIPT"
assert_stderr "no args prints error"         "no input file"  "$ESCRIPT"
assert_exit   "unknown flag exits 1"         1  "$ESCRIPT" --bogus
assert_exit   "missing file exits 1"         1  "$ESCRIPT" /nonexistent/points.txt
assert_exit   "empty file exits 2"           2  "$ESCRIPT" "$EMPTY"

# ── summary ─────────────────────────────────────────────────────────────────
summary
