#!/usr/bin/env bash
# Launch a hyperparameter sweep using the Pegasus Ensemble Manager.
# Each config file in configs/exp_*.yml becomes a separate workflow
# in the ensemble, running in parallel up to MAX_RUNNING.
#
# Usage:
#   ./run_sweep.sh [--max-running 4]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_RUNNING="${1:-4}"
ENSEMBLE_NAME="fl_sweep_$(date +%Y%m%d_%H%M%S)"

echo "=== FL Hyperparameter Sweep ==="
echo "  Ensemble: ${ENSEMBLE_NAME}"
echo "  Max concurrent: ${MAX_RUNNING}"

# Start the ensemble manager service (idempotent if already running)
pegasus-em server &
sleep 2

# Create the ensemble with throttling
pegasus-em create "${ENSEMBLE_NAME}" \
    -R "max_running=${MAX_RUNNING}" \
    -R "max_planning=2"

# Submit each experiment config as a workflow in the ensemble
for config in "${SCRIPT_DIR}"/configs/exp_*.yml; do
    config_name="$(basename "${config}" .yml)"
    echo "  Submitting: ${config_name}"

    pegasus-em submit "${ENSEMBLE_NAME}.${config_name}" \
        "${SCRIPT_DIR}/plan_fl_workflow.sh" --config "${config}"
done

echo ""
echo "Sweep submitted. Monitor with:"
echo "  pegasus-em workflows ${ENSEMBLE_NAME}"
echo "  pegasus-em status ${ENSEMBLE_NAME}.<workflow_name>"
