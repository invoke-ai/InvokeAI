# InvokeAI Web UI

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [InvokeAI Web UI](#invokeai-web-ui)
  - [Core Libraries](#core-libraries)
    - [Redux Toolkit](#redux-toolkit)
    - [Socket\.IO](#socketio)
    - [Chakra UI](#chakra-ui)
    - [KonvaJS](#konvajs)
    - [Vite](#vite)
    - [i18next & Weblate](#i18next--weblate)
    - [openapi-typescript](#openapi-typescript)
    - [reactflow](#reactflow)
    - [zod](#zod)
  - [Client Types Generation](#client-types-generation)
  - [Package Scripts](#package-scripts)
  - [Contributing](#contributing)
    - [Dev Environment](#dev-environment)
      - [VSCode Remote Dev](#vscode-remote-dev)
    - [Production builds](#production-builds)

<!-- /code_chunk_output -->

The UI is a fairly straightforward Typescript React app.

## Core Libraries

InvokeAI's UI is made possible by a number of excellent open-source libraries. The most heavily-used are listed below, but there are many others.

### Redux Toolkit

[Redux Toolkit] is used for state management and fetching/caching:

- `RTK-Query` for data fetching and caching
- `createAsyncThunk` for a couple other HTTP requests
- `createEntityAdapter` to normalize things like images and models
- `createListenerMiddleware` for async workflows

We use [redux-remember] for persistence.

### Socket\.IO

[Socket.IO] is used for server-to-client events, like generation process and queue state changes.

### Chakra UI

[Chakra UI] is our primary UI library, but we also use a few components from [Mantine v6].

### KonvaJS

[KonvaJS] powers the canvas. In the future, we'd like to explore [PixiJS] or WebGPU.

### Vite

[Vite] is our bundler.

### i18next & Weblate

We use [i18next] for localization, but translation to languages other than English happens on our [Weblate] project. **Only the English source strings should be changed on this repo.**

### openapi-typescript

[openapi-typescript] is used to generate types from the server's OpenAPI schema. See TYPES_CODEGEN.md.

### reactflow

[reactflow] powers the Workflow Editor.

### zod

[zod] schemas are used to model data structures and provide runtime validation.

## Client Types Generation

We use [openapi-typescript] to generate types from the app's OpenAPI schema.

The generated types are written to `invokeai/frontend/web/src/services/api/schema.d.ts`. This file is committed to the repo.

The server must be started and available at <http://127.0.0.1:9090>.

```sh
# from the repo root, start the server
python scripts/invokeai-web.py
# from invokeai/frontend/web/, run the script
pnpm typegen
```

## Package Scripts

See `package.json` for all scripts.

Run with `pnpm <script name>`.

- `dev`: run the frontend in dev mode, enabling hot reloading
- `build`: run all checks (madge, eslint, prettier, tsc) and then build the frontend
- `typegen`: generate types from the OpenAPI schema (see [Client Types Generation](#client-types-generation))
- `lint:madge`: check frontend for circular dependencies
- `lint:eslint`: check frontend for code quality
- `lint:prettier`: check frontend for code formatting
- `lint:tsc`: check frontend for type issues
- `lint`: run all checks concurrently
- `fix`: run `eslint` and `prettier`, fixing fixable issues

## Contributing

Thanks for your interest in contributing to the InvokeAI Web UI!

We encourage you to ping @psychedelicious and @blessedcoolant on [discord] if you want to contribute, just to touch base and ensure your work doesn't conflict with anything else going on. The project is very active.

### Dev Environment

Install [node] and [pnpm].

From `invokeai/frontend/web/` run `pnpm i` to get everything set up.

Start everything in dev mode:

1. Start the dev server: `pnpm dev`
2. Start the InvokeAI Nodes backend: `python scripts/invokeai-web.py # run from the repo root`
3. Point your browser to the dev server address e.g. <http://localhost:5173/>

#### VSCode Remote Dev

We've noticed an intermittent issue with the VSCode Remote Dev port forwarding. If you use this feature of VSCode, you may intermittently click the Invoke button and then get nothing until the request times out. Suggest disabling the IDE's port forwarding feature and doing it manually via SSH:

`ssh -L 9090:localhost:9090 -L 5173:localhost:5173 user@host`

### Production builds

For a number of technical and logistical reasons, we need to commit UI build artefacts to the repo.

If you submit a PR, there is a good chance we will ask you to include a separate commit with a build of the app.

To build for production, run `pnpm build`.

[node]: https://nodejs.org/en/download/
[pnpm]: https://github.com/pnpm/pnpm
[discord]: https://discord.gg/ZmtBAhwWhy
[Redux Toolkit]: https://github.com/reduxjs/redux-toolkit
[redux-remember]: https://github.com/zewish/redux-remember
[Socket.IO]: https://github.com/socketio/socket.io
[Chakra UI]: https://github.com/chakra-ui/chakra-ui
[Mantine v6]: https://v6.mantine.dev/
[KonvaJS]: https://github.com/konvajs/react-konva
[PixiJS]: https://github.com/pixijs/pixijs
[Vite]: https://github.com/vitejs/vite
[i18next]: https://github.com/i18next/react-i18next
[Weblate]: https://hosted.weblate.org/engage/invokeai/
[openapi-typescript]: https://github.com/drwpow/openapi-typescript
[reactflow]: https://github.com/xyflow/xyflow
[zod]: https://github.com/colinhacks/zod
