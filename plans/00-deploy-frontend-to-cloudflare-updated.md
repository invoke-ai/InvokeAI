# Task: Deploy Frontend to Cloudflare

**Status:** In Progress

**Depends On:** None

---

### 1. Objective

To deploy the InvokeAI frontend to Cloudflare Pages, making it accessible via a public URL and configured to communicate with the backend API.

---

### 2. The "Why"

This is the first step in making the application publicly accessible. By deploying the frontend first, we can establish the user-facing part of the application and prepare for the backend integration. A successful frontend deployment will also allow us to test the user interface in a production-like environment.

---

### 3. The "How" - Implementation Details

*   **Files to be modified:**
    *   `invokeai/frontend/web/src/services/api/index.ts` (to configure the API base URL for production)
*   **New files to be created:**
    *   None

*   **Step-by-step instructions:**

    1.  **Configure API Base URL for Production:**
        *   In `invokeai/frontend/web/src/services/api/index.ts`, modify the `getBaseUrl` function to use the production backend URL when the application is running in a production environment. This will be achieved by checking for the `VITE_API_URL` environment variable.
        *   **Proposed Code Change:**
            ```typescript
            export const getBaseUrl = (): string => {
              if (import.meta.env.VITE_API_URL) {
                return import.meta.env.VITE_API_URL;
              }
              const baseUrl = $baseUrl.get();
              return baseUrl || window.location.href.replace(/\/$/, '');
            };
            ```

    2.  **Build the Frontend:**
        *   Navigate to the `invokeai/frontend/web` directory.
        *   Run `pnpm install` to install the dependencies.
        *   Run `pnpm run build` to create a production build of the frontend in the `dist` directory. This command will also run the linter to ensure code quality.

    3.  **Configure Cloudflare Pages:**
        *   Create a new Cloudflare Pages project.
        *   Connect the project to the GitHub repository.
        *   Configure the build settings:
            *   **Build command:** `cd invokeai/frontend/web && pnpm install && pnpm run build`
            *   **Build output directory:** `invokeai/frontend/web/dist`
        *   Add an environment variable to the Cloudflare Pages project:
            *   **Variable name:** `VITE_API_URL`
            *   **Value:** The URL of the deployed backend API.

    4.  **Deploy to Cloudflare:**
        *   Trigger a deployment in the Cloudflare Pages dashboard.
        *   Verify that the deployment is successful and the frontend is accessible at the Cloudflare Pages URL.
        *   Test the application to ensure that it can successfully communicate with the backend API.

---

### 4. Acceptance Criteria

*   [x] The `getBaseUrl` function in `invokeai/frontend/web/src/services/api/index.ts` is updated to use the production backend URL.
*   [x] The frontend is successfully built without errors.
*   [ ] The Cloudflare Pages project is configured correctly with the necessary build settings and environment variables.
*   [ ] The frontend is deployed to Cloudflare Pages and is accessible via a public URL.
*   [ ] The deployed frontend can successfully communicate with the backend API.

---

### 5. Notes and Discoveries

*   **AI Assistant (Cline) - 2025-07-28:** The `getBaseUrl` function in `invokeai/frontend/web/src/services/api/index.ts` has been successfully modified to use the `VITE_API_URL` environment variable for production builds.
*   **AI Assistant (Cline) - 2025-07-28:** The frontend dependencies were installed using `pnpm install`, and the production build was successfully created using `pnpm run build`. The build process completed without any errors, and the `dist` directory is ready for deployment.
*   The `vite.config.mts` file contains a proxy configuration for the development server. This configuration will not be used in the production deployment on Cloudflare Pages. The frontend will communicate directly with the backend API using the URL specified in the `VITE_API_URL` environment variable.
*   The frontend is built with Vite, React, and TypeScript.
*   State management is handled by Redux Toolkit and Nanostores.
*   The UI is built with Chakra UI and a custom component library.
*   API communication is handled by Redux Toolkit Query.

---

### 6. Next Steps

*   [ ] Once the frontend is deployed and tested, we will proceed with the backend refactoring and deployment as outlined in the subsequent task plans.
