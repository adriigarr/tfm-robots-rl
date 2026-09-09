#!/bin/bash
# E1.5 SUBWP — solo s42 training (recuperación tras fallo inicial)
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"
WORLD_SUBWP="$WORLDS_DIR/warehouse_1_subwp_1ped.wbt"

echo "======================================================="
echo " SUBWP E1.5 s42 — training (recuperación)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

echo "e1_5_s42" > "$SUBWP_DIR/current_stage.txt"
"$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_SUBWP"

echo "[OK] SUBWP E1.5 s42 training completado — $(date '+%Y-%m-%d %H:%M:%S')"
