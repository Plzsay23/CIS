#!/usr/bin/env bash
set -euo pipefail

pkill -TERM -f "/home/lerobot/CIS/dashboard/dashboard_v2_server.py" 2>/dev/null || true
pkill -TERM -f "dashboard_v2_server.py" 2>/dev/null || true
sleep 1
pkill -KILL -f "/home/lerobot/CIS/dashboard/dashboard_v2_server.py" 2>/dev/null || true
pkill -KILL -f "dashboard_v2_server.py" 2>/dev/null || true
echo "[dashboard_v2] stopped"
