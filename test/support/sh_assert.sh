#!/usr/bin/env bash
# Shared assertions for the shell-level CLI suites (cli_test.sh, escript_test.sh).
# Source this file, then call the assert_* helpers; finish with `summary`.

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

assert_no_stderr() {
  local desc="$1"
  shift
  local err
  err=$("$@" 2>&1 >/dev/null || true)
  if [ -z "$err" ]; then pass "$desc"; else fail "$desc (unexpected stderr: $err)"; fi
}

# Exits non-zero if anything failed, so the script can be used as a CI step.
summary() {
  echo ""
  echo "Results: $PASS passed, $FAIL failed"
  [ "$FAIL" -eq 0 ]
}
