# State Management Guide

This frontend uses two complementary state layers:

- **Redux Toolkit store** for durable, persisted, or undoable application state that participates in middleware, RTK
  Query caching, or needs to be visible in devtools.
- **Nanostores atoms** for lightweight, imperative values that behave like configuration flags or transient UI helpers.

Keeping the contract for each layer explicit makes it easier for both humans and AI agents to add or refactor state
without guessing.

## When to reach for Redux

Use a slice when any of the following are true:

- The value must survive reloads via `redux-remember`, feed undo/redo history, or plug into listener middleware.
- Multiple features consume the value through selectors/memoization and we want time-travel/debug tooling support.
- The state is derived from API responses or dispatches actions that other middleware reacts to (e.g. queue, gallery,
  canvas, workflow, parameters, system).

### Adding a slice

Redux slices live in `features/<feature>/store`. When adding one:

1. Create the slice with `createSlice`, a Zod schema, and an exported `SliceConfig<T>`.
2. Call `registerSlice(yourSliceConfig);` inside the registration block in `app/store/store.ts`. The reducer map is
   built automatically (undoable slices remain wrapped for you).
3. Decide which keys should persist and update the slice’s `persistConfig` denylist/migration accordingly.
4. Expose selectors in the slice file so callers avoid reaching into `RootState` manually.

> Tip: Because registration is manual, double-check both `SLICE_CONFIGS` and `ALL_REDUCERS` before opening a PR—missing
> one of them will compile but crash on rehydration.

## When to prefer nanostores

Nanostores excel for simple, ephemeral state that is driven by host configuration or narrow feature surface areas (e.g.
modal toggles, auth tokens, injected UI overrides).

Choose a nanostore when:

- The value does **not** need to persist, participate in undo, or trigger Redux middleware.
- Only a handful of components or hooks care about it, often within a single feature module.
- Updating it synchronously from `useEffect` or external callbacks is simpler than dispatching actions.

### Pattern

We generally pair an atom with hooks:

```ts
const initialState = { isOpen: false };
const $example = atom(initialState);

export const useExampleState = () => useStore($example);
export const useExampleApi = () =>
  useMemo(
    () => ({
      open: () => $example.set({ isOpen: true }),
      close: () => $example.set(initialState),
    }),
    []
  );
```

Keep nano state colocated with the feature (`features/<feature>/store/state.ts`) so it is easy to discover. When the
state graduates to something that needs persistence or devtools visibility, migrate it into a Redux slice and remove the
atom.

## Decision checklist

| Question                                                             | Use Redux | Use nanostore |
| -------------------------------------------------------------------- | --------- | ------------- |
| Needs undo/history or persistence?                                   | ✅        | ❌            |
| Needs to trigger listener middleware / RTK Query matchers?           | ✅        | ❌            |
| Only toggles a modal or stores host-provided config?                 | ❌        | ✅            |
| Debuggable via Redux devtools or inspectable across features needed? | ✅        | ❌            |

If the answers are mixed, favour Redux—the manual boilerplate is worth the consistency.

## Additional notes

- Keep prop-to-atom syncing (`InvokeAIUI`, host integration) inside `useEffect` hooks so atoms behave like runtime
  configuration switches.
- Avoid mixing direct Redux access and nanostore state for the same concern; pick one abstraction per capability.
- When deprecating a slice, clean up its registrations in `app/store/store.ts` and update `RootState` consumers before
  deleting the file. The `changeBoardModal` modal is a good reference for migrating a simple slice to nanostores.

This guide should help both humans and AI collaborators make consistent choices and reduce churn when the application’s
state model evolves.
