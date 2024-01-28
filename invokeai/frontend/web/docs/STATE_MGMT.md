# State Management

The app makes heavy use of Redux Toolkit, its Query library, and `nanostores`.

## Redux

TODO

## `nanostores`

[nanostores] is a tiny state management library. It provides both imperative and declarative APIs.

### Example

```ts
export const $myStringOption = atom<string | null>(null);

// Outside a component, or within a callback for performance-critical logic
$myStringOption.get();
$myStringOption.set('new value');

// Inside a component
const myStringOption = useStore($myStringOption);
```

### Where to put nanostores

- For global application state, export your stores from `invokeai/frontend/web/src/app/store/nanostores/`.
- For feature state, create a file for the stores next to the redux slice definition (e.g. `invokeai/frontend/web/src/features/myFeature/myFeatureNanostores.ts`).
- For hooks with global state, export the store from the same file the hook is in, or put it next to the hook.

### When to use nanostores

- For non-serializable data that needs to be available throughout the app, use `nanostores` instead of a global.
- For ephemeral global state (i.e. state that does not need to be persisted), use `nanostores` instead of redux.
- For performance-critical code and in callbacks, redux selectors can be problematic due to the declarative reactivity system. Consider refactoring to use `nanostores` if there's a **measurable** performance issue.

[nanostores]: https://github.com/nanostores/nanostores/
