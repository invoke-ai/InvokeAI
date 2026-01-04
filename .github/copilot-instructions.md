# InvokeAI Copilot Instructions

## Project Overview

InvokeAI is a leading creative engine built to empower professionals and enthusiasts alike. It's a full-featured AI-assisted image generation environment designed for creatives and enthusiasts, with an industry-leading web-based UI. The project serves as the foundation for multiple commercial products and is free to use under a commercially-friendly license.

**Key Technologies:**
- Backend: Python 3.11-3.12, FastAPI, Socket.IO, PyTorch
- Frontend: React, TypeScript, Vite, Redux
- AI/ML: Stable Diffusion (SD1.5, SD2.0, SDXL, FLUX), Diffusers, Transformers
- Database: SQLite
- Package Management: uv (backend), pnpm (frontend)

## Repository Structure

```
invokeai/
├── app/                    # Main application code
│   ├── api/               # FastAPI routes and API endpoints
│   ├── invocations/       # Node-based invocation system
│   └── services/          # Core services (model management, image storage, etc.)
├── backend/               # AI/ML core functionality
│   ├── image_util/        # Image processing utilities
│   ├── model_management/  # Model loading and management
│   └── stable_diffusion/  # SD pipeline implementations
├── frontend/web/          # React web UI
│   └── src/
│       ├── app/           # App setup and configuration
│       ├── common/        # Shared utilities and types
│       ├── features/      # Feature-specific components and logic
│       └── services/      # API clients and services
├── configs/               # Configuration files
└── tests/                 # Test suite
```

## Development Environment Setup

### Prerequisites
- Python 3.11 or 3.12 (as specified in pyproject.toml: `>=3.11, <3.13`)
- Node.js v22.14.0 or compatible v22.x LTS version (see .nvmrc)
- pnpm v10.x (minimum v10 required, see package.json)
- Git LFS
- uv (Python package manager)

### Initial Setup

1. **Clone and configure Git LFS:**
   ```bash
   git clone https://github.com/invoke-ai/InvokeAI.git
   cd InvokeAI
   git config lfs.fetchinclude "*"
   git lfs pull
   ```

2. **Backend Setup:**
   ```bash
   # Install Python dependencies with dev extras (adjust --python version as needed: 3.11 or 3.12)
   uv pip install -e ".[dev,test,docs,xformers]" --python 3.12 --python-preference only-managed --index=https://download.pytorch.org/whl/cu128 --reinstall
   ```

3. **Frontend Setup:**
   ```bash
   cd invokeai/frontend/web
   pnpm install
   pnpm build  # For production build
   # OR
   pnpm dev    # For development mode (hot reload on localhost:5173)
   ```

4. **Database:** Use an ephemeral in-memory database for development by setting `use_memory_db: true` and `scan_models_on_startup: true` in your `invokeai.yaml` file.

### Common Development Commands

**Backend:**
```bash
make ruff              # Run ruff linter and formatter
make ruff-unsafe       # Run ruff with unsafe fixes
make mypy              # Run type checker
make test              # Run unit tests
pytest tests/          # Run fast tests only
pytest tests/ -m slow  # Run slow tests
```

**Frontend:**
```bash
cd invokeai/frontend/web
pnpm lint              # Run all linters
pnpm lint:eslint       # Check ESLint issues
pnpm lint:prettier     # Check formatting
pnpm lint:tsc          # Check TypeScript issues
pnpm fix               # Auto-fix issues
pnpm test:no-watch     # Run tests
```

**Documentation:**
```bash
make docs              # Serve mkdocs with live reload
mkdocs serve           # Alternative command
```

## Code Style and Conventions

### Python (Backend)

**Style Guidelines:**
- Use **Ruff** for linting and formatting (replaces Black, isort, flake8)
- Line length: 120 characters
- Type hints are required (mypy strict mode with Pydantic plugin)
- Use absolute imports (no relative imports allowed)
- Follow PEP 8 conventions

**Key Conventions:**
- All invocations must inherit from `BaseInvocation`
- Use the `@invocation` decorator for invocation classes
- Invocation class names should end with "Invocation" (e.g., `ResizeImageInvocation`)
- Use `InputField()` for invocation inputs and `OutputField()` for outputs
- All invocations must have a docstring
- Services should provide an abstract base class interface

**Import Style:**
```python
# Use absolute imports from invokeai
from invokeai.invocation_api import BaseInvocation, invocation, InputField
from invokeai.app.services.image_records.image_records_common import ImageCategory
```

**Example Invocation:**
```python
from invokeai.invocation_api import (
    BaseInvocation,
    invocation,
    InputField,
    OutputField,
)

@invocation('my_invocation', title='My Invocation', tags=['image'], category='image')
class MyInvocation(BaseInvocation):
    """Does something with an image."""
    
    image: ImageField = InputField(description="The input image")
    width: int = InputField(default=512, description="Output width")
    
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Implementation
        pass
```

### TypeScript/JavaScript (Frontend)

**Style Guidelines:**
- Use **ESLint** and **Prettier** for linting and formatting
- Prefer TypeScript over JavaScript
- Use functional components with hooks
- Use Redux Toolkit for state management
- Colocate tests with source files using `.test.ts` suffix

**Key Conventions:**
- Tests should use Vitest
- No tests needed for trivial code (type definitions, re-exports)
- UI tests are not currently implemented
- Keep components focused and composable

**Import Organization:**
```typescript
// External imports first
import { useCallback } from 'react';
import { useDispatch } from 'react-redux';

// Internal app imports
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { AppDispatch } from 'app/store/store';
```

## Architecture

### Backend Architecture

**Invocation System:**
- **Invocations**: Modular nodes that represent single operations with inputs and outputs
- **Sessions**: Maintain graphs of linked invocations and execution history
- **Invoker**: Manages sessions and the invocation queue
- **Services**: Provide functionality to invocations (model management, image storage, etc.)

**Key Principles:**
- Invocations form directed acyclic graphs (no loops)
- All invocations are auto-discovered from `invokeai/app/invocations/`
- Services use abstract base classes for flexibility
- Applications interact through the invoker, not directly with core code

### Frontend Architecture

**State Management:**
- Redux Toolkit for global state
- Feature-based organization
- Slices for different app areas (ui, gallery, generation, etc.)

**API Communication:**
- REST API via FastAPI
- Real-time updates via Socket.IO
- OpenAPI-generated TypeScript types

## Testing Practices

### Backend Testing

**Test Organization:**
- All tests in `tests/` directory, mirroring `invokeai/` structure
- Use pytest with markers: `@pytest.mark.slow` for tests >1s
- Default: fast tests only (`-m "not slow"`)
- Coverage target: 85%

**Test Commands:**
```bash
pytest tests/              # Fast tests
pytest tests/ -m slow      # Slow tests
pytest tests/ -m ""        # All tests
pytest tests/ --cov        # With coverage report
```

**Model Testing:**
- Auto-download models if not present
- Avoid re-downloading existing models
- Reuse models across tests when possible
- Use fixtures: `model_installer`, `torch_device`

### Frontend Testing

**Test Guidelines:**
- Use Vitest for unit tests
- Colocate tests with source files (`.test.ts`)
- No UI/integration tests currently
- Skip tests for trivial code

## Common Tasks

### Adding a New Invocation

1. Create a new file in `invokeai/app/invocations/`
2. Define class inheriting from `BaseInvocation`
3. Add `@invocation` decorator with unique ID
4. Define inputs with `InputField()`
5. Implement `invoke()` method
6. Return appropriate output type
7. Add to `__init__.py` in the invocations directory

### Adding a New Service

1. Create abstract base class interface in `invokeai/app/services/`
2. Implement default local implementation
3. Register service in invoker setup
4. Avoid loading heavy dependencies unless implementation is used

### Frontend Development

1. Make changes in `invokeai/frontend/web/src/`
2. Run linters: `pnpm lint`
3. Fix issues: `pnpm fix`
4. Test in dev mode: `pnpm dev` (localhost:5173)
5. Build for production: `pnpm build`

### Updating OpenAPI Types

When backend API changes:
```bash
cd invokeai/frontend/web
python ../../../scripts/generate_openapi_schema.py | pnpm typegen
```

## Build and Deployment

**Backend Build:**
```bash
# Build wheel
cd scripts && ./build_wheel.sh
```

**Frontend Build:**
```bash
make frontend-build
# OR
cd invokeai/frontend/web && pnpm build
```

**Running the Application:**
```bash
invokeai-web  # Starts server on localhost:9090
```

## Contributing Guidelines

1. **Before starting:** Check in with maintainers to ensure alignment with project vision
2. **Development:**
   - Fork and clone the repository
   - Create a feature branch
   - Make changes following style guidelines
   - Add/update tests as needed
   - Run linters and tests
3. **Pull Requests:**
   - Use the PR template
   - Provide clear summary and QA instructions
   - Link related issues (use "Closes #123" to auto-close)
   - Check all items in the PR checklist
   - Update documentation if needed
   - Update migration if redux slice changes
4. **Code Review:** Be responsive to feedback and ready to iterate

## Important Notes

- **Database Migrations:** Redux slice changes require corresponding migrations
- **Python Linting/Formatting:** The project uses **Ruff** for new code (via `make ruff`), which replaces black, flake8, and isort. However, pre-commit hooks still reference the older tools - this is a known transition state.
- **Model Management:** Models are auto-registered on startup if configured
- **External Code:** Some directories contain external code (mediapipe_face, mlsd, normal_bae, etc.) and are excluded from linting
- **Platform Support:** Cross-platform (Linux, macOS, Windows) with GPU support (CUDA, ROCm)
- **Localization:** UI supports 20+ languages via Weblate

## Resources

- [Documentation](https://invoke-ai.github.io/InvokeAI/)
- [Discord Community](https://discord.gg/ZmtBAhwWhy)
- [GitHub Issues](https://github.com/invoke-ai/InvokeAI/issues)
- [Contributing Guide](https://invoke-ai.github.io/InvokeAI/contributing/)
- [Architecture Overview](docs/contributing/ARCHITECTURE.md)
- [Invocations Guide](docs/contributing/INVOCATIONS.md)
