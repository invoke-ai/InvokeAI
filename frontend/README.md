# Stable Diffusion Web UI

Demo at https://peaceful-otter-7a427f.netlify.app/ (not connected to back end)

## Test and Build

from root/frontend:

-   `yarn dev` runs `tsc` in a watch mode, which runs `vite build` when `tsc` is successful

from root:

-   `python backend/server.py` from project root to serve both frontend and backend at http://localhost:9090

## TODO

-   Search repo for "TODO"
-   My one gripe with Chakra: no way to disable all animations right now and drop the dependence on `framer-motion`. I would prefer to save the ~30kb on bundle and have zero animations. This is on the Chakra roadmap. See https://github.com/chakra-ui/chakra-ui/pull/6368 for last discussion on this. Need to check in on this issue periodically.
