#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 <task> <clear|random> [--resets N] [--output-root DIR] [options]"
    exit 0
fi
if [[ "$#" -lt 2 ]]; then
    echo "Provide a task and a scene setting. See $0 --help." >&2
    exit 2
fi
task=$1
setting=$2
shift 2
export EMBODICHAIN_DATA_ROOT="${EMBODICHAIN_DATA_ROOT:-$repo_root/assets}"
export EMBODICHAIN_SIM_EXIT_PROCESS=0
if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-offscreen}"
    export SDL_VIDEO_DRIVER="${SDL_VIDEO_DRIVER:-offscreen}"
fi
cd "$repo_root"
exec uv run --locked --python 3.12 python -m scripts.visualize_distribution \
    "$task" "$setting" "$@"
