#!/bin/sh
# Playable demo: fake Elegoo in the Doom room + the full agent + dashboard at http://localhost:8000.
# Ctrl-C stops everything. Logs: demo_logs/. The agent is restarted automatically if it ever exits.
cd "$(dirname "$0")"
PORT_V=${PORT_V:-8081}; PORT_C=${PORT_C:-8100}; mkdir -p demo_logs
.venv/bin/python fake_elegoo.py --seed ${SEED:--1} --seconds 0 --video-port $PORT_V --cmd-port $PORT_C > demo_logs/fake.log 2>&1 &
FAKE=$!; trap 'kill $FAKE 2>/dev/null; pkill -f "main.py --host 127.0.0.1" 2>/dev/null; exit 0' INT TERM EXIT
sleep 2
if [ "${RELAY:-0}" = "1" ]; then              # VAST via the event VM: open a tunnel to the dashboard and print the VM command
  export MEMORY_BACKEND=relay RELAY_TOKEN=$(python3 -c "import secrets;print(secrets.token_hex(16))")
  cloudflared tunnel --url http://localhost:8000 --no-autoupdate > demo_logs/tunnel.log 2>&1 &
  TUN=$!; trap 'kill $FAKE $TUN 2>/dev/null; pkill -f "main.py --host 127.0.0.1" 2>/dev/null; exit 0' INT TERM EXIT
  for i in $(seq 1 30); do URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' demo_logs/tunnel.log | head -1); [ -n "$URL" ] && break; sleep 1; done
  CMD="curl -s '$URL/relay/script?token=$RELAY_TOKEN' -o vm_relay.py && python3 vm_relay.py $URL $RELAY_TOKEN"
  printf '%s' "$CMD" | pbcopy; echo "$CMD" > demo_logs/vm_command.txt
  echo "Run on the event VM (copied to clipboard):"; echo "  $CMD"
fi
while true; do
  # output goes to a file inside the wrapped command, so the key wrapper never buffers it
  CATBOT_SIM_URL=http://localhost:$PORT_V \
  COSMOS_BASE=${COSMOS_BASE:-http://166.19.38.112:8001/v1} COSMOS_MODEL=${COSMOS_MODEL:-nvidia/cosmos3-nano-reasoner} \
  MEMORY_BACKEND=$MEMORY_BACKEND RELAY_TOKEN=$RELAY_TOKEN claude-secrets run --timeout 86400 \
    --inject NVIDIA_API_KEY=NVIDIA_API_KEY --inject WANDB_API_KEY=WANDB_API_KEY --inject COSMOS_API_KEY=COSMOS_API_KEY -- \
    sh -c "exec .venv/bin/python main.py --host 127.0.0.1 --video-port $PORT_V --cmd-port $PORT_C --dash --seconds 0 --every 3 >> demo_logs/agent.log 2>&1" > /dev/null 2>&1
  echo "$(date '+%H:%M:%S') agent exited; restarting in 2 s" >> demo_logs/agent.log
  sleep 2
done
