import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import type { LibrarySyncStatus } from './libraryAutosave';

export const workflowLibrarySyncStore = createExternalStore<{ status: LibrarySyncStatus }>({ status: 'idle' });

registerAccountOwnedResource({
  clear: () => {
    workflowLibrarySyncStore.setSnapshot({ status: 'idle' });
  },
  name: 'workflow-library-sync',
});

export const setWorkflowLibrarySyncStatus = (status: LibrarySyncStatus): void => {
  workflowLibrarySyncStore.setSnapshot({ status });
};
