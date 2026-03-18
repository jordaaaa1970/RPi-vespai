#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="vespai-web.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
RUN_USER="$(id -un)"
RUN_GROUP="$(id -gn)"

if [ ! -x "$SCRIPT_DIR/start_vespai_web.sh" ]; then
    chmod +x "$SCRIPT_DIR/start_vespai_web.sh"
fi

echo "Installing ${SERVICE_NAME} for project: ${SCRIPT_DIR}"

sudo tee "$SERVICE_PATH" >/dev/null <<EOF
[Unit]
Description=VespAI Web Dashboard
After=network.target

[Service]
Type=simple
PermissionsStartOnly=true
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${SCRIPT_DIR}
ExecStartPre=/usr/bin/mkdir -p ${SCRIPT_DIR}/logs
ExecStartPre=/usr/bin/touch ${SCRIPT_DIR}/logs/vespai.log
ExecStartPre=/usr/bin/chown ${RUN_USER}:${RUN_GROUP} ${SCRIPT_DIR}/logs/vespai.log
Environment=PYTHONPATH=/usr/lib/python3/dist-packages
ExecStart=${SCRIPT_DIR}/start_vespai_web.sh --motion
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "Installed and started ${SERVICE_NAME}."
systemctl status "$SERVICE_NAME" --no-pager -n 20 || true
