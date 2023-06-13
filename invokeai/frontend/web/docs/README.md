# InvokeAI Web UI

- [InvokeAI Web UI](#invokeai-web-ui)
  - [Stack](#stack)
  - [Contributing](#contributing)
    - [Dev Environment](#dev-environment)
    - [Production builds](#production-builds)

The UI is a fairly straightforward Typescript React app. The only really fancy stuff is the Unified Canvas.

Code in `invokeai/frontend/web/` if you want to have a look.

## Stack

State management is Redux via [Redux Toolkit](https://github.com/reduxjs/redux-toolkit). We lean heavily on RTK:
- `createAsyncThunk` for HTTP requests
- `createEntityAdapter` for fetching images and models
- `createListenerMiddleware` for workflows

The API client and associated types are generated from the OpenAPI schema. See API_CLIENT.md.

Communication with server is a mix of HTTP and [socket.io](https://github.com/socketio/socket.io-client) (with a simple socket.io redux middleware to help).

[Chakra-UI](https://github.com/chakra-ui/chakra-ui) for components and styling.

[Konva](https://github.com/konvajs/react-konva) for the canvas, but we are pushing the limits of what is feasible with it (and HTML canvas in general). We plan to rebuild it with [PixiJS](https://github.com/pixijs/pixijs) to take advantage of WebGL's improved raster handling.

[Vite](https://vitejs.dev/) for bundling.

Localisation is via [i18next](https://github.com/i18next/react-i18next), but translation happens on our [Weblate](https://hosted.weblate.org/engage/invokeai/) project. Only the English source strings should be changed on this repo.

## Contributing

Thanks for your interest in contributing to the InvokeAI Web UI!

We encourage you to ping @psychedelicious and @blessedcoolant on [Discord](https://discord.gg/ZmtBAhwWhy) if you want to contribute, just to touch base and ensure your work doesn't conflict with anything else going on. The project is very active.

### Dev Environment

Install [node](https://nodejs.org/en/download/) and [yarn classic](https://classic.yarnpkg.com/lang/en/).

From `invokeai/frontend/web/` run `yarn install` to get everything set up.

Start everything in dev mode:

1. Start the dev server: `yarn dev`
2. Start the InvokeAI Nodes backend: `python scripts/invokeai-web.py # run from the repo root`
3. Point your browser to the dev server address e.g. <http://localhost:5173/>

#### VSCode Remote Dev

We've noticed an intermittent issue with the VSCode Remote Dev port forwarding. If you use this feature of VSCode, you may intermittently click the Invoke button and then get nothing until the request times out. Suggest disabling the IDE's port forwarding feature and doing it manually via SSH:

`ssh -L 9090:localhost:9090 -L 5173:localhost:5173 user@host`

### Production builds

For a number of technical and logistical reasons, we need to commit UI build artefacts to the repo.

If you submit a PR, there is a good chance we will ask you to include a separate commit with a build of the app.

To build for production, run `yarn build`.
