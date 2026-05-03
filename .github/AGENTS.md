# Agent Instructions

## Package Management

This project uses **pnpm** exclusively for package management in the frontend (`invokeai/frontend/web/`).

- ✅ Use `pnpm` commands (e.g., `pnpm install`, `pnpm run`)
- ❌ Never use `npm` or `yarn` commands
- ❌ Never suggest creating or using `package-lock.json` or `yarn.lock`
- ✅ The lock file is `pnpm-lock.yaml`

Use the following pnpm commands for typical operations:

- pnpm -C invokeai/frontend/web install
- pnpm -C invokeai/frontend/web build
- pnpm -C invokeai/frontend/web lint:tsc
- pnpm -C invokeai/frontend/web lint:dpdm
- pnpm -C invokeai/frontend/web lint:eslint
- pnpm -C invokeai/frontend/web lint:prettier

## Project Structure

- Backend: Python in `invokeai/`
- Frontend: TypeScript/React in `invokeai/frontend/web/` (uses pnpm)
