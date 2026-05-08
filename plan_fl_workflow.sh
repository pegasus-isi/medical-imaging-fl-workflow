#!/usr/bin/env bash
# Plan and submit a single FL workflow.
# Used standalone or by the Ensemble Manager for parameter sweeps.
#
# Usage:
#   ./plan_fl_workflow.sh --config configs/default.yml
#   ./plan_fl_workflow.sh --config configs/exp_e1_baseline.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/configs/default.yml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

CONFIG_NAME="$(basename "${CONFIG}" .yml)"
SUBMIT_DIR="${SCRIPT_DIR}/work/submit/${CONFIG_NAME}"

echo "=== FL Workflow: ${CONFIG_NAME} ==="
echo "  Config: ${CONFIG}"
echo "  Submit dir: ${SUBMIT_DIR}"

# Generate workflow + catalogs
cd "${SCRIPT_DIR}"
python3 fl_main.py --config "${CONFIG}" --output "fl_main_${CONFIG_NAME}.yml" --plan

# Plan with Pegasus (CondorIO mode)
pegasus-plan \
    --conf pegasus.properties \
    --sites condorpool \
    --output-sites local \
    --dir "${SUBMIT_DIR}" \
    --cleanup inplace \
    --submit \
    "fl_main_${CONFIG_NAME}.yml"
