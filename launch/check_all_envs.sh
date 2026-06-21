#!/bin/bash
# ----------------------------------------------------------------------------
# check_all_envs.sh — 依次运行所有任务环境的冒烟检查 (random + clear)
#
# 用法:
#   ./launch/check_all_envs.sh [extra_args...]
#
# 示例:
#   ./launch/check_all_envs.sh
#   ./launch/check_all_envs.sh --max_episodes 1
# ----------------------------------------------------------------------------

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

EXTRA_ARGS=("$@")

# 排除 other_tasks 的所有任务
TASKS=(
    click_bell
    drawer_open_place
    handle_basket
    item_assembly
    items_handover
    manipulate_pipette
    mixer_operating
    sample_loading
    table_rearrangement
    water_pouring
)

TOTAL=$(( ${#TASKS[@]} * 2 ))
COUNT=0
PASSED=0
FAILED=0
FAILED_LIST=()

echo "========================================="
echo "  RoboSynChallenge — 全环境冒烟检查"
echo "  模式: random + clear"
echo "  任务数: ${#TASKS[@]} x 2 = $TOTAL"
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

        echo -ne "[${COUNT}/${TOTAL}] \033[1;34mRUN\033[0m   $TASK ($SETTING) ... "

        LOG_FILE="/tmp/check_env_${TASK}_${SETTING}_$(date +%s).log"
        if python -m scripts.run_env \
            --gym_config "$GYM_CONFIG" \
            --action_config "$ACTION_CONFIG" \
            --num_envs 1 \
            --headless \
            --filter_dataset_saving \
            --max_episodes 1 \
            "${EXTRA_ARGS[@]}" \
            > "$LOG_FILE" 2>&1; then
            echo -e "\033[1;32mOK\033[0m"
            PASSED=$((PASSED + 1))
            rm -f "$LOG_FILE"
        else
            echo -e "\033[1;31mFAIL\033[0m  (log: $LOG_FILE)"
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$TASK ($SETTING)")
        fi

        sleep 2
    done
done

echo ""
echo "========================================="
echo "  结果: $PASSED/$TOTAL 通过, $FAILED 失败"
echo "========================================="

if [ $FAILED -gt 0 ]; then
    echo -e "\033[1;31m失败任务:\033[0m"
    for t in "${FAILED_LIST[@]}"; do
        echo "  - $t"
    done
    echo ""
    exit 1
else
    echo -e "\033[1;32m全部任务通过！\033[0m"
    echo ""
    exit 0
fi
