Here is a detailed report on the frontend stack of the InvokeAI application:

### __1. Build Tool: Vite__

Yes, the frontend uses __Vite__ as its build tool. This is confirmed by the presence of `vite.config.mts` and the `scripts` in `package.json`, which use `vite` for development (`dev`), building (`build`), and previewing (`preview`).

### __2. Core Framework: React__

The application is built using __React__. This is evident from the `react` and `react-dom` dependencies in `package.json` and the use of `@vitejs/plugin-react-swc` in the `vite.config.mts` file.

### __3. Language: TypeScript__

The frontend is written in __TypeScript__. This is indicated by the presence of `tsconfig.json`, the `.tsx` file extension for the main entry point (`src/main.tsx`), and the use of TypeScript-related packages like `@types/react`, `@typescript-eslint/parser`, and `typescript` in the `devDependencies`.

### __4. State Management: Redux Toolkit & Nanostores__

The application uses a combination of __Redux Toolkit__ and __Nanostores__ for state management:

- __Redux Toolkit:__ The presence of `@reduxjs/toolkit` and `react-redux` indicates that Redux is used for managing the application's state. The API service is also built on top of Redux Toolkit Query.
- __Nanostores:__ The use of `@nanostores/react` and the import of `$authToken`, `$baseUrl`, and `$projectId` from `app/store/nanostores` in `services/api/index.ts` show that Nanostores is used for managing smaller, more isolated pieces of state.

### __5. UI Library: Chakra UI & Custom Components__

The frontend uses a combination of a UI library and custom components:

- __Chakra UI:__ The `chakra-react-select` dependency suggests that Chakra UI is used for some UI components.
- __@invoke-ai/ui-library:__ The application also uses a custom UI library, `@invoke-ai/ui-library`, which likely contains a set of reusable components specific to the InvokeAI application.

### __6. API Communication: Redux Toolkit Query__

As mentioned earlier, the application uses __Redux Toolkit Query__ for data fetching and caching. This provides a robust and efficient way to interact with the backend API.

### __7. Key Dependencies__

Here are some other key dependencies that provide insight into the frontend's capabilities:

- __`@xyflow/react`:__ A library for building node-based UIs, which is likely used for the graph/node editor.
- __`socket.io-client`:__ Used for real-time communication with the backend, likely for things like progress updates and notifications.
- __`konva`:__ A 2D canvas library that is likely used for the image editor.
- __`i18next`:__ A library for internationalization and localization.
- __`zod`:__ A TypeScript-first schema declaration and validation library, used for data validation.

### __8. Development and Tooling__

- __Package Manager:__ The project uses `pnpm` as its package manager.
- __Linting and Formatting:__ The project uses __ESLint__ for linting and __Prettier__ for code formatting to maintain code quality and consistency.
- __Testing:__ The project uses __Vitest__ for unit and integration testing.
- __Storybook:__ The project uses __Storybook__ for developing and documenting UI components in isolation.

### __Summary__

The InvokeAI frontend is a modern, well-structured web application built with a robust and scalable tech stack. The use of Vite, React, TypeScript, Redux Toolkit, and a custom UI library provides a solid foundation for building a complex and feature-rich user interface.
