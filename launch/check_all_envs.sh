#!/bin/bash
# ----------------------------------------------------------------------------
# check_all_envs.sh — all environment smoke tests (random + clear)
#
# usage:
#   ./launch/check_all_envs.sh [--episode-timeout DURATION] [extra_args...]
#
# sample
#   ./launch/check_all_envs.sh
#   ./launch/check_all_envs.sh --episode-timeout 5m
#   ./launch/check_all_envs.sh --max_episodes 1
#
# DURATION supports the GNU timeout format (for example: 90s, 3m, 1h).
# The default timeout for each environment test is 3 minutes.
# ----------------------------------------------------------------------------

set -e
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

TEST_START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

GPU_INFO="unavailable"
if command -v nvidia-smi >/dev/null 2>&1; then
    if GPU_OUTPUT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null); then
        GPU_INFO="${GPU_OUTPUT%%$'\n'*}"
        if [ -z "$GPU_INFO" ]; then
            GPU_INFO="unavailable"
        fi
    fi
fi

EPISODE_TIMEOUT="${EPISODE_TIMEOUT:-3m}"
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --episode-timeout)
            if [ "$#" -lt 2 ]; then
                echo "Error: --episode-timeout requires a duration (for example: 3m)." >&2
                exit 2
            fi
            EPISODE_TIMEOUT="$2"
            shift 2
            ;;
        --episode-timeout=*)
            EPISODE_TIMEOUT="${1#*=}"
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$EPISODE_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?[smhd]?$ ]]; then
    echo "Error: invalid --episode-timeout value '$EPISODE_TIMEOUT'." >&2
    echo "Use a duration such as 180s, 3m, or 1h." >&2
    exit 2
fi

if ! command -v timeout >/dev/null 2>&1; then
    echo "Error: GNU timeout is required but was not found." >&2
    exit 2
fi

IN_EPISODE_SCREEN=0

enter_episode_screen() {
    # Use the terminal's alternate screen so episode logs do not overwrite the
    # main smoke-test progress. Non-interactive runs continue to log inline.
    if [ ! -t 1 ]; then
        return
    fi

    IN_EPISODE_SCREEN=1
    if command -v tput >/dev/null 2>&1; then
        tput smcup 2>/dev/null || printf '\033[?1049h'
        tput clear 2>/dev/null || printf '\033[2J\033[H'
    else
        printf '\033[?1049h\033[2J\033[H'
    fi
}

leave_episode_screen() {
    if [ "$IN_EPISODE_SCREEN" -ne 1 ]; then
        return
    fi

    if command -v tput >/dev/null 2>&1; then
        tput rmcup 2>/dev/null || printf '\033[?1049l'
    else
        printf '\033[?1049l'
    fi
    IN_EPISODE_SCREEN=0
}

trap 'leave_episode_screen' EXIT
trap 'leave_episode_screen; exit 130' INT
trap 'leave_episode_screen; exit 143' TERM

TASKS=(
    click_bell
    drawer_open_place
    water_pouring
    table_rearrangement
    handle_basket
    items_handover
    manipulate_pipette
    mixer_operating
    sample_loading
    item_assembly
)

TOTAL=$(( ${#TASKS[@]} * 2 ))
COUNT=0
PASSED=0
FAILED=0
TIMED_OUT=0
FAILED_LIST=()

echo "========================================="
echo "  RoboSynChallenge — All Environment Smoke Test"
echo "  Model: random + clear"
echo "  Number of tasks: ${#TASKS[@]} x 2 = $TOTAL"
echo "  Timeout per test: $EPISODE_TIMEOUT"
echo "========================================="
echo ""

for TASK in "${TASKS[@]}"; do
    for SETTING in random clear; do
        COUNT=$((COUNT + 1))

        GYM_CONFIG="configs/${TASK}/${SETTING}/gym_config.json"
        if [ -f "configs/${TASK}/action_config.json" ]; then
            ACTION_CONFIG="configs/${TASK}/action_config.json"
        else
            ACTION_CONFIG="configs/${TASK}/${SETTING}/action_config.json"
        fi

        if [ ! -f "$GYM_CONFIG" ] || [ ! -f "$ACTION_CONFIG" ]; then
            echo -e "[${COUNT}/${TOTAL}] \033[1;33mSKIP\033[0m  $TASK ($SETTING) — 配置文件缺失"
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$TASK ($SETTING): missing config")
            continue
        fi

        echo -e "[${COUNT}/${TOTAL}] \033[1;34mRUN\033[0m   $TASK ($SETTING)"

        LOG_FILE="/tmp/check_env_${TASK}_${SETTING}_$(date +%s).log"
        enter_episode_screen
        echo -e "\033[1;34mEpisode [${COUNT}/${TOTAL}]: $TASK ($SETTING)\033[0m"
        echo "Timeout: $EPISODE_TIMEOUT"
        echo "========================================="

        if timeout --signal=TERM --kill-after=10s "$EPISODE_TIMEOUT" \
            python -m scripts.run_env \
                --gym_config "$GYM_CONFIG" \
                --action_config "$ACTION_CONFIG" \
                --num_envs 1 \
                --headless \
                --max_episodes 1 \
                "${EXTRA_ARGS[@]}" \
                2>&1 | tee "$LOG_FILE"; then
            EPISODE_STATUS=0
        else
            EPISODE_STATUS=$?
        fi

        leave_episode_screen

        if [ "$EPISODE_STATUS" -eq 0 ]; then
            echo -e "\033[1;32mOK\033[0m"
            PASSED=$((PASSED + 1))
            rm -f "$LOG_FILE"
        elif [ "$EPISODE_STATUS" -eq 124 ] || [ "$EPISODE_STATUS" -eq 137 ]; then
            echo -e "\033[1;33mTIMEOUT\033[0m  (limit: $EPISODE_TIMEOUT, log: $LOG_FILE)"
            FAILED=$((FAILED + 1))
            TIMED_OUT=$((TIMED_OUT + 1))
            FAILED_LIST+=("$TASK ($SETTING): timeout after $EPISODE_TIMEOUT")
        else
            echo -e "\033[1;31mFAIL\033[0m  (exit: $EPISODE_STATUS, log: $LOG_FILE)"
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$TASK ($SETTING)")
        fi

        sleep 2
    done
done

echo ""
echo "============================================================"
echo "  Smoke Test Verification"
echo "  Started (UTC): $TEST_START_UTC"
echo "  GPU: $GPU_INFO"
echo "  Result: $PASSED/$TOTAL passed, $FAILED failed ($TIMED_OUT timed out)"
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    echo -e "\033[1;31mFailed tasks list:\033[0m"
    for t in "${FAILED_LIST[@]}"; do
        echo "  - $t"
    done
    echo ""
    exit 1
else
    echo -e "\033[1;32mAll tasks passed!\033[0m"
    echo ""
    exit 0
fi
