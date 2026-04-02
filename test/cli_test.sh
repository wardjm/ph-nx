#!/usr/bin/env bash
# Integration tests for the mix ph_nx CLI.
# Runs the mix task directly so no escript build is required.
set -euo pipefail

export MIX_ENV=test

PASS=0
FAIL=0

pass() { echo "PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "FAIL: $1"; FAIL=$((FAIL + 1)); }

assert_exit() {
  local desc="$1" expected="$2"
  shift 2
  local actual
  actual=0
  "$@" >/dev/null 2>&1 || actual=$?
  if [ "$actual" -eq "$expected" ]; then pass "$desc"; else fail "$desc (expected exit $expected, got $actual)"; fi
}

assert_stdout() {
  local desc="$1" pattern="$2"
  shift 2
  local out
  out=$("$@" 2>/dev/null)
  if echo "$out" | grep -q "$pattern"; then pass "$desc"; else fail "$desc (pattern '$pattern' not found in output)"; fi
}

assert_stderr() {
  local desc="$1" pattern="$2"
  shift 2
  local err
  err=$("$@" 2>&1 >/dev/null || true)
  if echo "$err" | grep -q "$pattern"; then pass "$desc"; else fail "$desc (pattern '$pattern' not found in stderr)"; fi
}

TMPDIR_CUSTOM=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR_CUSTOM"; }
trap cleanup EXIT

SQUARE="$TMPDIR_CUSTOM/square.txt"
cat > "$SQUARE" <<'EOF'
# unit square
0.0,0.0
1.0,0.0
1.0,1.0
0.0,1.0
EOF

EMPTY="$TMPDIR_CUSTOM/empty.txt"
cat > "$EMPTY" <<'EOF'
# only comments

EOF

TABS="$TMPDIR_CUSTOM/tabs.txt"
printf '0.0\t0.0\n1.0\t0.0\n1.0\t1.0\n0.0\t1.0\n' > "$TABS"

INVALID="$TMPDIR_CUSTOM/invalid.txt"
printf '0.0,0.0\n1.0,bad\n' > "$INVALID"

# ── help / usage ────────────────────────────────────────────────────────────
assert_exit   "--help exits 0"               0       mix ph_nx --help
assert_stdout "--help prints Usage"          "Usage" mix ph_nx --help

# ── argument errors ─────────────────────────────────────────────────────────
assert_exit   "no args exits non-zero"       1       mix ph_nx
assert_stderr "no args prints error"         "no input file" mix ph_nx

assert_exit   "unknown flag exits non-zero"  1       mix ph_nx --bogus
assert_stderr "unknown flag prints error"    "Invalid option" mix ph_nx --bogus

assert_exit   "multiple files exits non-zero" 1      mix ph_nx a.txt b.txt
assert_stderr "multiple files prints error"  "too many" mix ph_nx a.txt b.txt

# ── file errors ─────────────────────────────────────────────────────────────
assert_exit   "missing file exits non-zero"  1       mix ph_nx /nonexistent/points.txt
assert_stderr "missing file prints error"    "Error reading" mix ph_nx /nonexistent/points.txt

assert_exit   "empty file exits 2"           2       mix ph_nx "$EMPTY"
assert_stderr "empty file prints friendly error" "no data points" mix ph_nx "$EMPTY"

assert_exit   "invalid coords exits 2"       2       mix ph_nx "$INVALID"
assert_stderr "invalid coords names the line" "line 2" mix ph_nx "$INVALID"

# ── successful runs ─────────────────────────────────────────────────────────
assert_exit   "valid file exits 0"           0       mix ph_nx "$SQUARE"
assert_stdout "valid file prints point count" "4 points" mix ph_nx "$SQUARE"
assert_stdout "valid file prints Betti numbers" "Betti" mix ph_nx "$SQUARE"

assert_exit   "tab-separated file exits 0"   0       mix ph_nx "$TABS"
assert_stdout "tab-separated prints point count" "4 points" mix ph_nx "$TABS"

assert_exit   "--max-dim option accepted"     0       mix ph_nx "$SQUARE" --max-dim 1
assert_exit   "--threshold option accepted"   0       mix ph_nx "$SQUARE" --threshold 5.0

# ── summary ─────────────────────────────────────────────────────────────────
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
