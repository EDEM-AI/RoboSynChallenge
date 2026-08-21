#!/bin/bash
# ----------------------------------------------------------------------------
# Optional helper for users who only have RoboSynChallenge checked out.
# It clones LeRobot inside policy/smolvla/lerobot and installs it editable in
# the current Python environment.
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${1:-$SCRIPT_DIR/lerobot}"
LEROBOT_REPO="${LEROBOT_REPO:-https://github.com/huggingface/lerobot.git}"
LEROBOT_REF="${LEROBOT_REF:-main}"

if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo "LeRobot already exists: $INSTALL_DIR"
else
    git clone "$LEROBOT_REPO" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
git fetch origin "$LEROBOT_REF"
git checkout "$LEROBOT_REF"
python -m pip install -U pip
python -m pip install -e .

echo "LeRobot is ready at: $INSTALL_DIR"
echo "For eval, set:"
echo "  export SMOLVLA_LEROBOT_ROOT=$INSTALL_DIR"
