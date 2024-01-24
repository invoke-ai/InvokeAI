# Invoke UI

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Invoke UI](#invoke-ui)
  - [Core Libraries](#core-libraries)
  - [Package Scripts](#package-scripts)
    - [Client Types Generation](#client-types-generation)
  - [Contributing](#contributing)
    - [Localization](#localization)
    - [Dev Environment](#dev-environment)
    - [VSCode Remote Dev](#vscode-remote-dev)

<!-- /code_chunk_output -->

## Core Libraries

Invoke's UI is made possible by a number of excellent open-source libraries. The most heavily-used are listed below, but there are many others.

- [Redux Toolkit]
- [redux-remember]
- [Socket.IO]
- [Chakra UI]
- [KonvaJS]
- [Vite]
- [openapi-typescript]
- [reactflow]
- [zod]

## Package Scripts

See `package.json` for all scripts.

Run with `pnpm <script name>`.

- `dev`: run the frontend in dev mode, enabling hot reloading
- `build`: run all checks (madge, eslint, prettier, tsc) and then build the frontend
- `typegen`: generate types from the OpenAPI schema (see [Client Types Generation])
- `lint:madge`: check frontend for circular dependencies
- `lint:eslint`: check frontend for code quality
- `lint:prettier`: check frontend for code formatting
- `lint:tsc`: check frontend for type issues
- `lint`: run all checks concurrently
- `fix`: run `eslint` and `prettier`, fixing fixable issues

### Client Types Generation

We use [openapi-typescript] to generate types from the app's OpenAPI schema.

The generated types are written to `invokeai/frontend/web/src/services/api/schema.d.ts`. This file is committed to the repo.

The server must be started and available at <http://127.0.0.1:9090>.

```sh
# from the repo root, start the server
python scripts/invokeai-web.py
# from invokeai/frontend/web/, run the script
pnpm typegen
```

## Contributing

Thanks for your interest in contributing to the Invoke Web UI!

We encourage you to ping @psychedelicious and @blessedcoolant on [discord] if you want to contribute, just to touch base and ensure your work doesn't conflict with anything else going on. The project is very active.

### Localization

We use [i18next] for localization, but translation to languages other than English happens on our [Weblate] project.

**Only the English source strings should be changed on this repo.**

### Dev Environment

Install [node] and [pnpm].

From `invokeai/frontend/web/` run `pnpm i` to get everything set up.

Start everything in dev mode:

1. From `invokeai/frontend/web/`: `pnpm dev`
2. From repo root: `python scripts/invokeai-web.py`
3. Point your browser to the dev server address e.g. <http://localhost:5173/>

### VSCode Remote Dev

We've noticed an intermittent issue with the VSCode Remote Dev port forwarding. If you use this feature of VSCode, you may intermittently click the Invoke button and then get nothing until the request times out.

We suggest disabling the IDE's port forwarding feature and doing it manually via SSH:

```sh
ssh -L 9090:localhost:9090 -L 5173:localhost:5173 user@host
```

[node]: https://nodejs.org/en/download/
[pnpm]: https://github.com/pnpm/pnpm
[discord]: https://discord.gg/ZmtBAhwWhy
[Redux Toolkit]: https://github.com/reduxjs/redux-toolkit
[redux-remember]: https://github.com/zewish/redux-remember
[Socket.IO]: https://github.com/socketio/socket.io
[Chakra UI]: https://github.com/chakra-ui/chakra-ui
[KonvaJS]: https://github.com/konvajs/react-konva
[Vite]: https://github.com/vitejs/vite
[i18next]: https://github.com/i18next/react-i18next
[Weblate]: https://hosted.weblate.org/engage/invokeai/
[openapi-typescript]: https://github.com/drwpow/openapi-typescript
[reactflow]: https://github.com/xyflow/xyflow
[zod]: https://github.com/colinhacks/zod
[Client Types Generation]: #client-types-generation
