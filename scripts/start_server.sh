#!/usr/bin/env bash
# ==============================================================================
# scripts/start_server.sh
#
# Clean, lifecycle-managed server starter for yasamarium/llmserver.
# Implements:
#   1. Process launch (uvicorn)
#   2. Health check verification (GET /health)
#   3. Automatic test request
#   4. Cloudflare tunnel exposure (Quick Tunnel or Named Tunnel)
#   5. ~5-hour execution monitoring
#   6. Graceful shutdown (SIGTERM with 30s timeout)
#   7. Periodic restart loop
# ==============================================================================

set -eo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-}"
RESTART_INTERVAL="${RESTART_INTERVAL_SECONDS:-18000}"  # 5 hours = 18000 seconds
HEALTH_TIMEOUT=120
RUN_SINGLE_CYCLE="${RUN_SINGLE_CYCLE:-false}"

echo "===================================================================="
echo " Starting yasamarium/llmserver Supervisor"
echo " Host: $HOST | Port: $PORT"
echo " Restart Interval: ${RESTART_INTERVAL}s (~$(( RESTART_INTERVAL / 3600 )) hours)"
echo " Single Cycle Mode: $RUN_SINGLE_CYCLE"
echo "===================================================================="

# Trap termination signals to ensure child processes are cleaned up
cleanup() {
    echo ""
    echo "Caught shutdown signal! Terminating server and tunnel..."
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Sending SIGTERM to server (PID: $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "$TUNNEL_PID" ] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo "Stopping tunnel (PID: $TUNNEL_PID)..."
        kill -TERM "$TUNNEL_PID" 2>/dev/null || true
    fi
    echo "Cleanup complete."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start Tunnel Helper
setup_tunnel() {
    # Check if cloudflared is installed
    if ! command -v cloudflared &> /dev/null; then
        echo "cloudflared not found. Checking if it can be downloaded..."
        if [ "$(uname)" = "Linux" ]; then
            echo "Downloading cloudflared binary for Linux x86_64..."
            curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
            chmod +x cloudflared
            export PATH="$PWD:$PATH"
        fi
    fi

    if command -v cloudflared &> /dev/null; then
        if [ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
            echo "Starting Cloudflare Named Tunnel using provided token..."
            cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN" > tunnel.log 2>&1 &
            TUNNEL_PID=$!
        else
            echo "Starting Cloudflare Quick Tunnel on port $PORT..."
            cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > tunnel.log 2>&1 &
            TUNNEL_PID=$!
            
            # Extract public URL from log
            echo "Waiting for public tunnel URL to appear in tunnel.log..."
            PUBLIC_URL=""
            for i in $(seq 1 30); do
                sleep 1
                if grep -o "https://[-a-zA-Z0-9@:%._\+~#=]\+\.trycloudflare\.com" tunnel.log > /dev/null 2>&1; then
                    PUBLIC_URL=$(grep -o "https://[-a-zA-Z0-9@:%._\+~#=]\+\.trycloudflare\.com" tunnel.log | head -n 1)
                    break
                fi
            done

            if [ -n "$PUBLIC_URL" ]; then
                echo ""
                echo "===================================================================="
                echo " [SUCCESS] Public Tunnel URL: $PUBLIC_URL"
                echo " Vercel app (yasamarium/llm) can connect to this endpoint."
                echo "===================================================================="
                echo ""
                
                # Automatically publish endpoint.txt to GitHub repo so Vercel frontend reads it directly
                echo "$PUBLIC_URL" > endpoint.txt
                git config user.name "github-actions[bot]"
                git config user.email "github-actions[bot]@users.noreply.github.com"
                git add endpoint.txt
                git commit -m "chore: update live endpoint.txt [skip ci]" || true
                git push origin main || true

                # If running inside GitHub Actions, publish to step summary
                if [ -n "$GITHUB_STEP_SUMMARY" ]; then
                    {
                        echo "## 🚀 yasamarium/llmserver is Online"
                        echo ""
                        echo "- **Public URL:** \`$PUBLIC_URL\`"
                        echo "- **Model:** \`qwen3-4b\`"
                        echo "- **Local Port:** \`$PORT\`"
                        echo ""
                        echo "### Example cURL Request:"
                        echo "\`\`\`bash"
                        echo "curl -X POST \"$PUBLIC_URL/v1/chat/completions\" \\"
                        echo "  -H \"Content-Type: application/json\" \\"
                        echo "  -H \"Authorization: Bearer \$API_KEY\" \\"
                        echo "  -d '{\"model\":\"qwen3-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":256}'"
                        echo "\`\`\`"
                    } >> "$GITHUB_STEP_SUMMARY"
                fi

                # Notify or update target repo if token is available
                if [ -f "scripts/update_endpoint.py" ]; then
                    python3 scripts/update_endpoint.py "$PUBLIC_URL" || true
                fi
            else
                echo "Warning: Could not automatically parse Quick Tunnel URL. See tunnel.log for details."
            fi
        fi
    else
        echo "Note: cloudflared is not installed; server is running locally on http://$HOST:$PORT"
    fi
}

cycle_count=0

while true; do
    cycle_count=$((cycle_count + 1))
    echo ""
    echo "--------------------------------------------------------------------"
    echo " [Cycle #$cycle_count] Starting server process at $(date -u)..."
    echo "--------------------------------------------------------------------"

    # 1. Start Server Process in background
    python3 -m uvicorn src.server:app --host "$HOST" --port "$PORT" &
    SERVER_PID=$!
    echo "Server process launched with PID: $SERVER_PID"

    # 2. Health check polling
    echo "Waiting for server to become healthy at http://127.0.0.1:$PORT/health..."
    healthy=false
    for (( i=1; i<=HEALTH_TIMEOUT; i++ )); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: Server process terminated prematurely during startup!"
            exit 1
        fi

        # Curl health endpoint
        if response=$(curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null); then
            if echo "$response" | grep -q '"status":\s*"ok"'; then
                echo "Server reports HEALTHY! Response: $response"
                healthy=true
                break
            fi
        fi
        sleep 2
    done

    if [ "$healthy" = false ]; then
        echo "ERROR: Server failed to report healthy within $HEALTH_TIMEOUT seconds."
        kill -9 "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi

    # 3. Setup tunnel (if first cycle)
    if [ -z "$TUNNEL_PID" ] || ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        setup_tunnel
    fi

    # 4. Run automatic self-test request
    echo "Running automatic test request to verify inference pipeline..."
    AUTH_HEADER=""
    if [ -n "$API_KEY" ]; then
        AUTH_HEADER="Authorization: Bearer $API_KEY"
    fi

    TEST_RES=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        ${AUTH_HEADER:+-H "$AUTH_HEADER"} \
        -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Ping test"}],"max_tokens":16}' || true)

    echo "Self-test response: $TEST_RES"

    # 5. Serve requests for RESTART_INTERVAL (~5 hours)
    echo "Serving requests for $RESTART_INTERVAL seconds (~$(( RESTART_INTERVAL / 3600 )) hours)..."
    
    elapsed=0
    interval_step=10
    while [ "$elapsed" -lt "$RESTART_INTERVAL" ]; do
        # Check if server process is still alive
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "WARNING: Server process died unexpectedly at elapsed=${elapsed}s!"
            break
        fi
        sleep "$interval_step"
        elapsed=$((elapsed + interval_step))
        
        # Periodic log heartbeat every 30 minutes
        if [ $((elapsed % 1800)) -eq 0 ]; then
            echo "Server heartbeat: active for $((elapsed / 60)) minutes. Next scheduled restart in $(( (RESTART_INTERVAL - elapsed) / 60 )) minutes."
        fi
    done

    # 6. Graceful restart sequence
    echo ""
    echo "Reached scheduled restart threshold (~$(( RESTART_INTERVAL / 3600 )) hours)."
    echo "Initiating graceful shutdown of server process (PID: $SERVER_PID)..."
    kill -TERM "$SERVER_PID" 2>/dev/null || true

    # Wait up to 30 seconds for graceful shutdown
    grace_counter=0
    while kill -0 "$SERVER_PID" 2>/dev/null && [ "$grace_counter" -lt 30 ]; do
        sleep 1
        grace_counter=$((grace_counter + 1))
    done

    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server did not exit within 30s. Sending SIGKILL..."
        kill -9 "$SERVER_PID" 2>/dev/null || true
    else
        echo "Server stopped cleanly."
    fi

    # Check if single cycle mode is requested
    if [ "$RUN_SINGLE_CYCLE" = "true" ]; then
        echo "Single cycle completed. Handing off to workflow supervisor."
        break
    fi

    echo "Restarting server in 5 seconds..."
    sleep 5
done
