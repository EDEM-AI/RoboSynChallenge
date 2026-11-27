#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec uv sync --project "$repo" --locked --python 3.12 "$@"
