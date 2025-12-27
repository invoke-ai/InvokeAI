# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InvokeAI is a professional creative AI tool for visual media generation, featuring a web-based UI and node-based workflow system. The project supports multiple Stable Diffusion variants (SD1.5, SD2.0, SDXL, SD3, FLUX, CogView4) and provides a complete application for image generation, editing, and workflow management.

**Tech Stack:**
- **Backend**: Python 3.11-3.12, FastAPI, SQLite, Pydantic
- **Frontend**: React 18, TypeScript, Vite, Chakra UI, Redux Toolkit, React Flow
- **AI/ML**: PyTorch 2.7.x, Diffusers, Transformers, ONNX
- **Package Management**: Python uses `uv` (preferred) or `pip`, Frontend uses `pnpm 10`

## Development Commands

### Python Backend

All Python commands should be run from the repository root.

**Environment Setup:**
```bash
# Create virtual environment (first time)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies (using uv - preferred)
uv pip install -e ".[test,dev]"
# or using pip
pip install -e ".[test,dev]"
```

**Development:**
```bash
# Run the application
invokeai-web
# or
python scripts/invokeai-web.py

# Linting and formatting
make ruff           # Fix and format code
make ruff-unsafe    # Fix all errors including unsafe fixes
make mypy           # Type checking
make mypy-all       # Type checking (all files)

# Testing
make test           # Run unit tests
pytest ./tests      # Run tests directly
pytest ./tests/path/to/test_file.py  # Run specific test file
pytest ./tests -m "not slow"         # Skip slow tests (default)
pytest ./tests -m ""                 # Run all tests including slow

# Documentation
make docs           # Serve mkdocs with live reload
```

**Frontend Type Generation:**
```bash
# Generate TypeScript types from OpenAPI schema
make frontend-typegen
# or manually:
cd invokeai/frontend/web
python ../../../scripts/generate_openapi_schema.py | pnpm typegen
```

**Build:**
```bash
make wheel          # Build distribution wheel
```

### Frontend

All frontend commands should be run from `invokeai/frontend/web/`.

```bash
# Install dependencies (first time or after package.json changes)
pnpm install

# Development
pnpm dev            # Start dev server at localhost:5173
pnpm dev:host       # Start dev server with network access

# Building
pnpm build          # Lint and build for production

# Linting and Testing
pnpm lint           # Run all linters (eslint, prettier, tsc, knip, dpdm)
pnpm lint:eslint    # ESLint only
pnpm lint:prettier  # Prettier formatting check
pnpm lint:tsc       # TypeScript type checking
pnpm lint:knip      # Check for unused dependencies
pnpm lint:dpdm      # Check for dependency cycles
pnpm fix            # Auto-fix linting and formatting issues

pnpm test           # Run tests with watch mode
pnpm test:no-watch  # Run tests once
pnpm test:ui        # Run tests with coverage UI

# Storybook
pnpm storybook      # Run Storybook dev server on :6006
```

## Architecture

### Backend Architecture

**Service-Oriented Design:**
The backend follows a service-oriented architecture with dependency injection. Core services are located in `invokeai/app/services/`:

- **Session Queue** (`session_queue/`): Manages execution queues for workflow sessions
- **Session Processor** (`session_processor/`): Executes invocations from the session queue
- **Invocation Cache** (`invocation_cache/`): Caches invocation results
- **Model Manager** (`model_manager/`): Handles model loading, registry, and metadata
- **Model Install** (`model_install/`): Manages model installation and downloads
- **Image Records/Files** (`image_records/`, `image_files/`): Image storage and metadata
- **Board Management** (`boards/`, `board_records/`, `board_images/`): Image organization
- **Workflow Records** (`workflow_records/`): Workflow persistence
- **Events** (`events/`): Event bus for real-time updates via Socket.IO

**Graph Execution System:**
The core execution engine is a node-based graph system (`invokeai/app/services/shared/README.md`):
- **Graph**: Immutable author-time workflow definition (nodes + edges)
- **GraphExecutionState**: Runtime scheduler that expands iterators and tracks execution
- **Indegree-based scheduling**: Nodes execute when all prerequisites complete
- **Class-grouped batching**: Nodes of the same type are executed together for efficiency
- Iterator/Collector patterns for loops and aggregations

**Invocations (Nodes):**
All nodes inherit from `BaseInvocation` in `invokeai/app/invocations/baseinvocation.py`:
- Located in `invokeai/app/invocations/`
- Custom nodes can be added in `invokeai/app/invocations/custom_nodes/`
- Each invocation is a Pydantic model with typed inputs/outputs
- Registered via `InvocationRegistry` for automatic discovery

**API Layer:**
FastAPI application in `invokeai/app/api_app.py` with routers in `invokeai/app/api/routers/`:
- RESTful HTTP endpoints for all operations
- Socket.IO for real-time events and progress updates
- OpenAPI schema auto-generated for frontend type generation

### Frontend Architecture

**Technology Stack:**
- **React 18** with TypeScript and Vite build system
- **State Management**: Redux Toolkit with Redux-Remember for persistence
- **UI Library**: Custom `@invoke-ai/ui-library` + Chakra UI components
- **Workflow Editor**: React Flow (@xyflow/react) for node-based workflows
- **Canvas**: Konva.js for the Unified Canvas feature
- **DnD**: Pragmatic Drag and Drop (@atlaskit)
- **API Client**: RTK Query auto-generated from OpenAPI schema

**Key Frontend Directories:**
- `src/app/`: Redux store configuration and app initialization
- `src/features/`: Feature modules (gallery, controlLayers, nodes, etc.)
- `src/services/api/`: Auto-generated API client from OpenAPI schema
- `src/common/`: Shared utilities, hooks, and components
- `src/theme/`: Chakra UI theme customization

**Frontend Build System:**
- Vite 7 for fast HMR and optimized production builds
- SWC for TypeScript compilation
- ESLint 9 with flat config for linting
- Vitest for unit testing
- Type generation from backend OpenAPI schema

**State Management Patterns:**
- RTK Query for server state (auto-synced with backend)
- Redux slices for client state (UI, selections, canvas state)
- Nanostores for lightweight reactive state outside Redux
- Local component state for ephemeral UI state

## Important Development Notes

### Python Development

**Type Checking:**
- MyPy is configured in strict mode (pyproject.toml)
- Many legacy modules are excluded (see `tool.mypy.overrides`)
- All new code should be fully typed

**Code Style:**
- Ruff for linting and formatting (replaces Black, isort, flake8)
- Line length: 120 characters
- Absolute imports only (no relative imports)

**Testing:**
- Pytest with coverage reporting
- Tests in `tests/` directory
- Markers: `@pytest.mark.slow` for long-running tests
- Target coverage: 85%

**Virtual Environment:**
The project uses a `.venv` directory. Dependencies are managed via `pyproject.toml` with optional extras:
- `[test]`: Testing dependencies (pytest, ruff, mypy)
- `[dev]`: Development tools (jurigged, pudb)
- `[docs]`: Documentation (mkdocs)
- `[xformers]`: xformers acceleration (CUDA only)
- Hardware-specific: `[cpu]`, `[cuda]`, `[rocm]`

### Frontend Development

**Code Organization:**
- Features are self-contained in `src/features/<feature-name>/`
- Each feature may have: components, hooks, store, types
- Shared code goes in `src/common/`

**Testing:**
- Tests colocated with code using `.test.ts` suffix
- Focus on logic, not trivial code or simple type definitions
- No UI snapshot tests currently

**API Integration:**
- Never manually write API client code
- Regenerate types when backend schema changes: `make frontend-typegen`
- API endpoints are auto-generated in `src/services/api/endpoints/`

**Common Gotchas:**
- Frontend dev server proxies `/api` requests to backend (configure in vite.config.ts)
- Backend must be running for full functionality during frontend development
- Socket.IO connection required for real-time updates

## Running the Full Stack

**Development Mode:**
1. Start backend: `invokeai-web` (from repo root with venv activated)
2. Start frontend: `cd invokeai/frontend/web && pnpm dev`
3. Access UI at http://localhost:5173 (frontend dev server)
4. Backend API at http://localhost:9090

**Production Mode:**
1. Build frontend: `cd invokeai/frontend/web && pnpm build`
2. Run backend: `invokeai-web`
3. Access at http://localhost:9090 (serves built frontend)

## Docker

Docker setup is in `docker/` directory:

```bash
cd docker
cp .env.sample .env
# Edit .env (set INVOKEAI_ROOT, GPU_DRIVER, etc.)
./run.sh
```

- **CUDA**: `GPU_DRIVER=cuda` (NVIDIA)
- **ROCm**: `GPU_DRIVER=rocm` (AMD)
- Runtime directory defaults to `~/invokeai`
- Models and outputs persist in `INVOKEAI_ROOT`

## Configuration

InvokeAI configuration is managed via:
1. `invokeai.yaml` in the root directory (or `INVOKEAI_ROOT`)
2. Environment variables (prefix `INVOKEAI_`)
3. CLI arguments (parsed in `invokeai/frontend/cli/arg_parser.py`)

Priority: CLI args > env vars > config file > defaults

See official docs for full configuration options: https://invoke-ai.github.io/InvokeAI/features/CONFIGURATION/

## Key Workflow Concepts

**Sessions & Queues:**
- Users create workflow sessions via the UI
- Sessions are queued and processed by the session processor
- Each session contains a graph of invocations to execute

**Boards & Gallery:**
- Images are organized into boards (like folders)
- Gallery provides search, filtering, and metadata viewing
- Images can be dragged onto node inputs for easy workflow building

**Model Management:**
- Models are installed via the Model Manager UI
- Supports Hugging Face repos, local files, and URLs
- Model metadata stored in SQLite (`invokeai_data/databases/`)

## Contributing Guidelines

1. **Backend changes**: Run `make ruff && make mypy` before committing
2. **Frontend changes**: Run `pnpm lint` before committing
3. **API changes**: Regenerate frontend types with `make frontend-typegen`
4. **New invocations**: Follow patterns in `invokeai/app/invocations/`
5. **Tests**: Add tests for new functionality, maintain 85% coverage
6. **Documentation**: Update relevant docs in `docs/` for user-facing changes

## Common Tasks

**Add a new invocation (node type):**
1. Create file in `invokeai/app/invocations/` (e.g., `my_node.py`)
2. Inherit from `BaseInvocation`, define inputs as fields
3. Define output type inheriting from `BaseInvocationOutput`
4. Implement `invoke()` method
5. Register with decorators: `@invocation()` and `@invocation_output()`
6. Rebuild frontend types: `make frontend-typegen`

**Add a new API endpoint:**
1. Create or modify router in `invokeai/app/api/routers/`
2. Follow FastAPI patterns for route definition
3. Update OpenAPI schema: `make openapi`
4. Regenerate frontend types: `make frontend-typegen`

**Debug the graph execution:**
- Enable debug logging in config
- Check session queue state via API
- Review `invokeai/app/services/shared/README.md` for graph execution details

**Add a new frontend feature:**
1. Create feature directory in `src/features/<feature-name>/`
2. Add Redux slice if needed in `store/`
3. Create components in `components/`
4. Add hooks in `hooks/`
5. Wire up to main app in `src/app/`
