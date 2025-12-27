#!/usr/bin/env bash
# InvokeAI-Meta Development Setup
# Gets InvokeAI running in development mode
clear
set -e

PROJECT_DIR="$INVOKE_DIR"
DATA_DIR="$INVOKE_DIR/invokeai_data"

echo "🚀 Setting up InvokeAI development environment..."

echo "$INVOKE_DIR"
echo "$PROJECT_DIR"

cd "$PROJECT_DIR"

# 1. Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --relocatable --prompt invoke-meta --python 3.12 --python-preference only-managed .venv
fi

# 2. Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# 3. Install InvokeAI in editable mode with dev dependencies
echo "Installing InvokeAI with dev dependencies (this may take a while)..."
uv pip install -e ".[dev,test,docs]" --python 3.12 --python-preference only-managed --torch-backend=cu128 --reinstall

# 4. Create data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory at $DATA_DIR..."
    mkdir -p "$DATA_DIR"
fi

# 5. Create initial config file
echo "Creating invokeai.yaml config..."
cat > "$DATA_DIR/invokeai.yaml" << EOF
# InvokeAI-Meta Configuration
# Data directory: $DATA_DIR

schema_version: 4.0.2

# Use persistent database (set to true for in-memory/ephemeral database)
use_memory_db: false

# Scan models on startup when using memory database
scan_models_on_startup: true

# Models directory - we'll symlink this to ComfyUI later
models_dir: $DATA_DIR/models

# Host and port
host: 0.0.0.0
port: 9090

# Log level
log_level: info
EOF

# 6. Install Node.js dependencies for frontend
echo "Installing frontend dependencies..."
cd invokeai/frontend/web
pnpm i

# 7. Build frontend (skip linting to avoid TypeScript crash)
echo "Building frontend (this may take a few minutes)..."
pnpm exec vite build

# 8. Install pypatchmatch
uv pip install pypatchmatch

cd "$PROJECT_DIR"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start InvokeAI:"
echo "  1. Activate venv: source .venv/bin/activate"
echo "  2. Run server: invokeai-web --root $DATA_DIR"
echo ""
echo "Server will be available at: http://127.0.0.1:9090"
echo ""
echo "For frontend development mode (with hot reload):"
echo "  cd invokeai/frontend/web && pnpm dev"
echo "  (Server will be at http://127.0.0.1:5173)"
