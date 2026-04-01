#!/usr/bin/env bash
# Wrapper so `mix ph_nx` can be run without specifying MIX_ENV.
# ex_doc (dev-only dep) requires erlang-parsetools which may not be installed.
MIX_ENV=test exec mix ph_nx "$@"
