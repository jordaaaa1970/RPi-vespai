#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="vespai-web.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

echo "Removing startup service: ${SERVICE_NAME}"

sudo systemctl disable --now "$SERVICE_NAME" 2>/dev/null || true
sudo rm -f "$SERVICE_PATH"
sudo systemctl daemon-reload
sudo systemctl reset-failed "$SERVICE_NAME" 2>/dev/null || true

echo "Removed ${SERVICE_NAME}."
