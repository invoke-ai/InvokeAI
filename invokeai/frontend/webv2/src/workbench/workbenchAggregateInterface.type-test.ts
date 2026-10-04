// @ts-expect-error Reducer actions are private; tests use workbenchState.testing explicitly.
import type { WorkbenchAction } from './workbenchState';
import type { WorkbenchInternalStore } from './workbenchStore';

// @ts-expect-error The reducer implementation is not exported under a public API name.
import { workbenchReducer } from './workbenchState';

declare const store: WorkbenchInternalStore;

// @ts-expect-error Raw reducer dispatch is not part of the aggregate interface.
store.dispatch({ type: 'createProject' });

// @ts-expect-error The public query snapshot does not expose the full aggregate state.
const fullState = store.queries.getSnapshot().state;

void store;
void fullState;
void (null as unknown as WorkbenchAction);
void workbenchReducer;
