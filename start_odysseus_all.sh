#!/bin/bash

# ---------- Configuration ----------
ODYSSEUS_DIR="$HOME/Desktop/Odyessus/odysseus"
SEARXNG_DIR="$HOME/Desktop/Odyessus/searxng"
UVICORN_HOST="127.0.0.1"
UVICORN_PORT="7000"
ROUTER_PORT="8001"
LOG_DIR="$ODYSSEUS_DIR/logs"
PRELOAD_MODEL="phi4-mini:3.8b"   # change to any small model you want pre‑loaded

# ---------- Global PIDs ----------
ROUTER_PID=""
PRELOAD_PID=""

# ---------- Functions ----------
check_port() {
    if lsof -i :$1 > /dev/null 2>&1; then
        echo "Port $1 is already in use."
        echo "Process:"
        lsof -i :$1
        read -p "Kill the process using port $1? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill -9 $(lsof -t -i :$1)
            echo "Process killed."
        else
            echo "Exiting. Free port $1 and try again."
            exit 1
        fi
    fi
}

start_searxng() {
    echo "Starting SearXNG..."
    cd "$SEARXNG_DIR" || { echo "SearXNG directory not found"; exit 1; }
    docker compose up -d
    echo "SearXNG started at http://localhost:8080"
    cd - > /dev/null
}

preload_model() {
    echo "Pre-loading $PRELOAD_MODEL (stays cached)..."
    # Run the model with an empty prompt, send to background
    ollama run "$PRELOAD_MODEL" <<< "" > /dev/null 2>&1 &
    PRELOAD_PID=$!
    # Give it a moment to load
    sleep 2
    if kill -0 $PRELOAD_PID 2>/dev/null; then
        echo "Model pre-loaded (PID: $PRELOAD_PID)"
    else
        echo "Warning: Model preload may have failed, but continuing..."
    fi
}

start_router() {
    echo "Starting router..."
    cd "$ODYSSEUS_DIR" || exit 1
    if [ ! -f "start_router.sh" ]; then
        echo "Router script not found – creating it"
        cat > start_router.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python -m uvicorn model_router:app --host 127.0.0.1 --port 8001
EOF
        chmod +x start_router.sh
    fi
    ./start_router.sh > "$LOG_DIR/router.log" 2>&1 &
    ROUTER_PID=$!
    echo $ROUTER_PID > "$LOG_DIR/router.pid"
    echo "Router started (PID: $ROUTER_PID)"
}

start_odysseus() {
    echo "Starting Odysseus main server..."
    cd "$ODYSSEUS_DIR" || exit 1
    source venv/bin/activate
    python -m uvicorn app:app --host "$UVICORN_HOST" --port "$UVICORN_PORT"
}

stop_all() {
    echo "Stopping all services..."
    # Stop router
    if [ -n "$ROUTER_PID" ] && kill -0 $ROUTER_PID 2>/dev/null; then
        kill $ROUTER_PID
        echo "Router stopped."
    fi
    if [ -f "$LOG_DIR/router.pid" ]; then
        kill $(cat "$LOG_DIR/router.pid") 2>/dev/null
        rm "$LOG_DIR/router.pid"
    fi
    # Stop preloaded model process (optional, model stays cached anyway)
    if [ -n "$PRELOAD_PID" ] && kill -0 $PRELOAD_PID 2>/dev/null; then
        kill $PRELOAD_PID
        echo "Preloaded model process stopped."
    fi
    # Stop SearXNG containers
    cd "$SEARXNG_DIR" && docker compose down > /dev/null 2>&1
    echo "All services stopped."
}

# ---------- Main ----------
mkdir -p "$LOG_DIR"

# Check port conflicts
check_port $UVICORN_PORT
check_port $ROUTER_PORT

# Start SearXNG if not already running
if ! docker ps --format '{{.Names}}' | grep -q "searxng"; then
    start_searxng
else
    echo "SearXNG already running."
fi

# Pre‑load a small model
preload_model

# Start router
start_router

# Trap Ctrl+C to clean up
trap stop_all INT

# Start Odysseus (foreground)
start_odysseus

# If Odysseus exits for any reason, clean up
stop_all