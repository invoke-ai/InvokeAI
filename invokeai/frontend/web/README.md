# InvokeAI UI dev setup

The UI is in `invokeai/frontend`.

## Environment set up

Install [node](https://nodejs.org/en/download/) (includes npm) and
[yarn](https://yarnpkg.com/getting-started/install).

From `invokeai/frontend/` run `yarn install --immutable` to get everything set up.

## Dev

1. Start the dev server: `yarn dev`
2. Start the InvokeAI UI per usual: `invokeai --web`
3. Point your browser to the dev server address e.g. `http://localhost:5173/`

To build for dev: `yarn build-dev`

To build for production: `yarn build`
