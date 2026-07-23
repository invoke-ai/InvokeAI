# Agent Instructions

## Package Management

This repository has two independently managed TypeScript/React frontends, both using **pnpm** exclusively:

- `invokeai/frontend/web/`
- `invokeai/frontend/webv2/`

- ✅ Use `pnpm` commands for both frontends.
- ✅ Run commands against the intended frontend with `pnpm -C <directory> ...`.
- ❌ Never use `npm` or `yarn`.
- ❌ Never create or use `package-lock.json` or `yarn.lock`.
- ✅ Each frontend owns its own `pnpm-lock.yaml`.

Typical commands for the established frontend:

- `pnpm -C invokeai/frontend/web install`
- `pnpm -C invokeai/frontend/web build`
- `pnpm -C invokeai/frontend/web lint:tsc`
- `pnpm -C invokeai/frontend/web lint:dpdm`
- `pnpm -C invokeai/frontend/web lint:eslint`
- `pnpm -C invokeai/frontend/web lint:prettier`

Typical commands for the Workbench frontend:

- `pnpm -C invokeai/frontend/webv2 install`
- `pnpm -C invokeai/frontend/webv2 build`
- `pnpm -C invokeai/frontend/webv2 format:check`
- `pnpm -C invokeai/frontend/webv2 lint:oxc`
- `pnpm -C invokeai/frontend/webv2 lint:tsc`
- `pnpm -C invokeai/frontend/webv2 test:all`

## Project Structure

- Backend: Python in `invokeai/`
- Established frontend: TypeScript/React in `invokeai/frontend/web/`
- Workbench frontend: TypeScript/React in `invokeai/frontend/webv2/`
