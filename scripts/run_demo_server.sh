#!/usr/bin/env bash
# Start / stop the PTIC web demo (internal research demo, not a medical device).
#
#   scripts/run_demo_server.sh start | stop | status | logs | link | code [on|off]
#
# `link` prints http://<address>:<port>/#code=<access code>: the page fills the code in from the part after # (which browsers
# never send to the server) and the browser remembers it. It shows the code, so run it in your own terminal, not on a shared screen.
#
# Binds to 127.0.0.1:8765 by default, so it is reachable only from this server (open it through an SSH tunnel, README "Web demo").
# Network mode, remembered between starts: logs/demo_server/host holds the address to bind (0.0.0.0 = every network interface)
# and logs/demo_server/access_code the access code (chmod 600). Both files are optional; PTIC_DEMO_HOST / PTIC_DEMO_PORT /
# PTIC_DEMO_TOKEN in the environment take precedence. Binding to anything but localhost requires an access code, unless the code
# is switched off on purpose with `code off` (it creates logs/demo_server/no_code; `code on` removes it and the saved code is used
# again). Without a code, anyone who can reach this server's address on the network can use the demo. Either takes effect at the
# next start.
# Other settings (upload limit, CORS, model, device) are listed at the top of src/demo/app.py.
set -euo pipefail
cd "$(dirname "$0")/.."
# Interpreter: $PTIC_PYTHON, else the project virtual environment .venv when it exists, else python3 on the PATH.
if [ -n "${PTIC_PYTHON:-}" ]; then PY=$PTIC_PYTHON; elif [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
RUN_DIR=logs/demo_server
PID_FILE=$RUN_DIR/server.pid
LOG=$RUN_DIR/server.log
HOST_FILE=$RUN_DIR/host
CODE_FILE=$RUN_DIR/access_code
NO_CODE_FILE=$RUN_DIR/no_code
if [ -z "${PTIC_DEMO_HOST:-}" ] && [ -f "$HOST_FILE" ]; then PTIC_DEMO_HOST=$(tr -d '[:space:]' < "$HOST_FILE"); fi
if [ -z "${PTIC_DEMO_TOKEN:-}" ] && [ -f "$CODE_FILE" ] && [ ! -f "$NO_CODE_FILE" ]; then PTIC_DEMO_TOKEN=$(tr -d '[:space:]' < "$CODE_FILE"); export PTIC_DEMO_TOKEN; fi
HOST=${PTIC_DEMO_HOST:-127.0.0.1}
PORT=${PTIC_DEMO_PORT:-8765}
# CPU thread count fixed before torch starts (changing it at run time can crash CPU ops in this PyTorch build; see src/common.py)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${PTIC_TORCH_THREADS:-8}} MKL_NUM_THREADS=${MKL_NUM_THREADS:-${PTIC_TORCH_THREADS:-8}}

running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
healthy() { "$PY" -I -c "import json,sys,urllib.request; json.load(urllib.request.urlopen('http://$1:$PORT/api/health', timeout=3))" 2>/dev/null; }
code_state() { "$PY" -I -c "import json,urllib.request; h = json.load(urllib.request.urlopen('http://$1:$PORT/api/health', timeout=3)); print('access code required' if h.get('token_required') else 'no access code')" 2>/dev/null; }
local_only() { [ "$HOST" = "127.0.0.1" ] || [ "$HOST" = "localhost" ]; }

case "${1:-}" in
  start)
    if running; then echo "already running (pid $(cat "$PID_FILE")) on port $PORT"; exit 0; fi
    if ! local_only && [ -z "${PTIC_DEMO_TOKEN:-}" ] && [ ! -f "$NO_CODE_FILE" ]; then
      echo "refusing to bind to $HOST without an access code: set PTIC_DEMO_TOKEN, keep the default 127.0.0.1, or switch the code off on purpose with: $0 code off"; exit 1
    fi
    mkdir -p "$RUN_DIR"
    echo "=== start $(date -u +%FT%TZ) host $HOST port $PORT" >> "$LOG"
    nohup "$PY" -I -m uvicorn src.demo.app:app --app-dir "$PWD" --host "$HOST" --port "$PORT" --workers 1 \
      --no-server-header >> "$LOG" 2>&1 &
    echo $! > "$PID_FILE"
    probe=$HOST; [ "$HOST" = "0.0.0.0" ] && probe=127.0.0.1
    for _ in $(seq 1 90); do
      if healthy "$probe"; then
        addr=$HOST; [ "$HOST" = "0.0.0.0" ] && addr=$(hostname -I | awk '{print $1}')
        echo "running (pid $(cat "$PID_FILE")): open http://$addr:$PORT/   log: $LOG"
        if [ -n "${PTIC_DEMO_TOKEN:-}" ]; then echo "an access code is required; for a link that fills it in, run: $PWD/scripts/run_demo_server.sh link"
        elif ! local_only; then echo "no access code: anyone who can reach $addr on the network can use the demo (to require the code again: $0 code on, then stop and start)"; fi
        exit 0
      fi
      if ! running; then echo "server exited during start-up; last log lines:"; tail -n 20 "$LOG"; rm -f "$PID_FILE"; exit 1; fi
      sleep 1
    done
    echo "started (pid $(cat "$PID_FILE")) but not answering yet; see $LOG"
    ;;
  stop)
    if ! running; then echo "not running"; rm -f "$PID_FILE"; exit 0; fi
    pid=$(cat "$PID_FILE"); kill "$pid"
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$pid" 2>/dev/null; then echo "pid $pid did not stop after 10 s; it is still running"; exit 1; fi
    rm -f "$PID_FILE"; echo "stopped"
    ;;
  status)
    if running; then
      probe=$HOST; [ "$HOST" = "0.0.0.0" ] && probe=127.0.0.1
      if healthy "$probe"; then echo "running and healthy (pid $(cat "$PID_FILE")) on $HOST:$PORT, $(code_state "$probe")"; else echo "process $(cat "$PID_FILE") alive but not answering"; fi
    else
      echo "not running"
    fi
    ;;
  logs) tail -n "${2:-50}" "$LOG" ;;
  link)
    addr=$HOST; [ "$HOST" = "0.0.0.0" ] && addr=$(hostname -I | awk '{print $1}')
    if [ -z "${PTIC_DEMO_TOKEN:-}" ]; then echo "http://$addr:$PORT/   (no access code is set)"; exit 0; fi
    code=$(printf '%s' "$PTIC_DEMO_TOKEN" | "$PY" -I -c "import sys, urllib.parse; print(urllib.parse.quote(sys.stdin.read(), safe=''))")
    echo "http://$addr:$PORT/#code=$code"
    echo "This link contains the access code: anyone who has it can use the demo."
    ;;
  code)
    case "${2:-}" in
      off) mkdir -p "$RUN_DIR"; touch "$NO_CODE_FILE"
           echo "access code switched off from the next start: anyone who can reach this server on the network will be able to use the demo" ;;
      on) rm -f "$NO_CODE_FILE"
          if [ -f "$CODE_FILE" ]; then echo "access code switched on from the next start (the saved code; old links work again)"
          else echo "access code switched on, but none is saved: put one in $CODE_FILE (chmod 600) before the next start"; fi ;;
      "") if [ -f "$NO_CODE_FILE" ]; then echo "next start: access code off ($NO_CODE_FILE exists)"; else echo "next start: access code on"; fi ;;
      *) echo "usage: $0 code [on|off]"; exit 2 ;;
    esac
    if running; then echo "the running server keeps its setting until: $0 stop && $0 start"; fi
    ;;
  *) echo "usage: $0 start|stop|status|logs [lines]|link|code [on|off]"; exit 2 ;;
esac
