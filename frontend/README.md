# Stable Diffusion Web UI

UI live at https://peaceful-otter-7a427f.netlify.app/

## Test and Build

from root/frontend:

-   `yarn dev` to run frontend only
-   `yarn build` to build frontend to `root/frontend/dist/`

from root:

-   `python backend/server.py` from project root to serve both frontend and backend at http://localhost:9090
-   `yarn --cwd frontend build && python backend/server.py` to build frontend and then serve everything
