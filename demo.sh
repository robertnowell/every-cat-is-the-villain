#!/bin/sh
# Playable demo: fake Elegoo in the Doom room + the full agent + dashboard at http://localhost:8000.
# Ctrl-C stops everything. Logs: demo_logs/. The agent is restarted automatically if it ever exits.
cd "$(dirname "$0")"
PORT_V=${PORT_V:-8081}; PORT_C=${PORT_C:-8100}; mkdir -p demo_logs
.venv/bin/python fake_elegoo.py --seed ${SEED:--1} --seconds 0 --video-port $PORT_V --cmd-port $PORT_C > demo_logs/fake.log 2>&1 &
FAKE=$!; trap 'kill $FAKE 2>/dev/null; pkill -f "main.py --host 127.0.0.1" 2>/dev/null; exit 0' INT TERM EXIT
sleep 2
while true; do
  # output goes to a file inside the wrapped command, so the key wrapper never buffers it
  CATBOT_SIM_URL=http://localhost:$PORT_V claude-secrets run --timeout 86400 \
    --inject NVIDIA_API_KEY=NVIDIA_API_KEY --inject WANDB_API_KEY=WANDB_API_KEY -- \
    sh -c "exec .venv/bin/python main.py --host 127.0.0.1 --video-port $PORT_V --cmd-port $PORT_C --dash --seconds 0 --every 3 >> demo_logs/agent.log 2>&1" > /dev/null 2>&1
  echo "$(date '+%H:%M:%S') agent exited; restarting in 2 s" >> demo_logs/agent.log
  sleep 2
done
