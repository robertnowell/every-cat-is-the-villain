#!/bin/bash
# The real car. Run from your own Terminal (macOS only lets Terminal reach the car's local network).
# Starts PAUSED: the wheels move only after Resume on the dashboard. Car on the floor, about 1 m clear around it.
# Dashboard: http://localhost:8000   Stop: Ctrl-C (the car stops when the link closes).
#   RELAY=1 ./run_car.sh   also sends clips to VAST through the event machine (uses phone data, ~1 GB/hour)
cd "$(dirname "$0")"; mkdir -p demo_logs
if [ "${RELAY:-0}" = "1" ]; then
  export MEMORY_BACKEND=relay RELAY_TOKEN=$(python3 -c "import secrets;print(secrets.token_hex(16))")
  cloudflared tunnel --url http://localhost:8000 --no-autoupdate > demo_logs/tunnel.log 2>&1 &
  TUN=$!
  for i in $(seq 1 30); do URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' demo_logs/tunnel.log | head -1); [ -n "$URL" ] && break; sleep 1; done
  echo "curl -fsS '$URL/relay/script?token=$RELAY_TOKEN' -o vm_relay.py && python3 vm_relay.py $URL $RELAY_TOKEN" > demo_logs/vm_command.txt
  echo "VAST relay: event-machine command saved to demo_logs/vm_command.txt"
fi
CS=$(command -v claude-secrets || ls ~/.claude/plugins/cache/claude-secrets-marketplace/claude-secrets/*/bin/claude-secrets 2>/dev/null | tail -1)
[ -x "$CS" ] || { echo "claude-secrets not found"; exit 1; }
echo "The car starts PAUSED. Open http://localhost:8000 and press Resume (or space) to let the wheels move. Ctrl-C stops everything."
: > demo_logs/car.log; tail -f demo_logs/car.log & TAIL=$!; trap 'kill $TAIL $TUN 2>/dev/null' EXIT
COSMOS_BASE=${COSMOS_BASE:-http://166.19.38.112:8001/v1} COSMOS_MODEL=${COSMOS_MODEL:-nvidia/cosmos3-nano-reasoner} \
MEMORY_BACKEND=$MEMORY_BACKEND RELAY_TOKEN=$RELAY_TOKEN \
"$CS" run --timeout 86400 --inject NVIDIA_API_KEY=NVIDIA_API_KEY --inject WANDB_API_KEY=WANDB_API_KEY --inject COSMOS_API_KEY=COSMOS_API_KEY -- \
  sh -c 'exec .venv/bin/python main.py --host 192.168.4.1 --dash --seconds 0 --every 3 "$@" >> demo_logs/car.log 2>&1' _ "$@" > /dev/null
