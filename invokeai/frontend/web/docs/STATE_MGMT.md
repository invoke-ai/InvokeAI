# State Management

The app makes heavy use of Redux Toolkit, its Query library, and `nanostores`.

## Redux

We use [Redux Toolkit] + RTK Query extensively.

### Persistence

The usual persistence layer for redux is [redux-persist]. Unfortunately, it is abandoned, and not possible to fork. Past releases of it depend on a malicious package that was removed from npm, so it's very difficult (impossible?) to build. The current state of the repo is also non-functional, as it was abandoned mid-rewrite.

We had a need to debounce our persistence, and patched redux-persist's build directly to do so. This didn't feel great. We've since moved to [redux-remember], a well-designed, minimal, and actively maintained library.

#### Slice migration

When rehydrating state, we sometimes need to migrate data. This is handled by the `unserialize` function in [store.ts], which is used by redux-remember to rehydrate persisted state. This function uses some lodash utils to strip out unknown keys, and merge any new keys into the rehydrated state.

Sometimes the shape of state changes, but it keeps the same property name in the slice. In that case, we need to _transform_ incoming data.

To handle this, each persisted slice must have a `SliceConfig`, which includes its latest initial value, and a migrate function. The migrate function, defined in the slice, does does any transformations and updates the version.

The version of the slice is currently only incremented when we need to run _transform_ migrations. If keys are added or removed from state, the version is not bumped.

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
[Redux Toolkit]: https://redux-toolkit.js.org/
[redux-persist]: https://github.com/rt2zz/redux-persist
[redux-remember]: https://github.com/zewish/redux-remember
[store.ts]: invokeai/frontend/web/src/app/store/store.ts
