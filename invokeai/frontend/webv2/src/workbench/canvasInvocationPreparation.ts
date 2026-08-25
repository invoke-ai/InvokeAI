import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { useExternalStoreSelector } from '@platform/state/selectors';

interface CanvasInvocationPreparationSnapshot {
  leases: ReadonlyMap<string, number>;
}

export interface CanvasInvocationPreparationLease {
  projectId: string;
  token: number;
}

const EMPTY_LEASES: ReadonlyMap<string, number> = new Map();
const store = createExternalStoreCore<CanvasInvocationPreparationSnapshot>({ leases: EMPTY_LEASES });
let nextLeaseToken = 1;

/**
 * Acquires the user-facing canvas preparation slot before any asynchronous
 * module loading, prompt expansion, persistence, compositing, or upload begins.
 * The deeper canvas orchestrator keeps its own guard for non-active entry
 * points; this one owns the Ctrl+Enter/topbar acknowledgement window.
 */
export const beginCanvasInvocationPreparation = (projectId: string): CanvasInvocationPreparationLease | null => {
  const { leases } = store.getSnapshot();

  if (leases.has(projectId)) {
    return null;
  }

  const lease = { projectId, token: nextLeaseToken };
  nextLeaseToken += 1;
  store.setSnapshot({ leases: new Map([...leases, [projectId, lease.token]]) });
  return lease;
};

export const endCanvasInvocationPreparation = (lease: CanvasInvocationPreparationLease): void => {
  const { leases } = store.getSnapshot();

  // Account invalidation clears every lease synchronously. If an old async
  // submission settles after a new account/project acquired the same id, its
  // stale token must not release the new owner's acknowledgement.
  if (leases.get(lease.projectId) !== lease.token) {
    return;
  }

  const nextLeases = new Map(leases);
  nextLeases.delete(lease.projectId);
  store.setSnapshot({ leases: nextLeases.size > 0 ? nextLeases : EMPTY_LEASES });
};

export const isCanvasInvocationPreparing = (projectId: string): boolean => store.getSnapshot().leases.has(projectId);

export const useIsCanvasInvocationPreparing = (projectId: string): boolean =>
  useExternalStoreSelector(store.subscribe, store.getSnapshot, (snapshot) => snapshot.leases.has(projectId));

registerAccountOwnedResource({
  clear: () => store.setSnapshot({ leases: EMPTY_LEASES }),
  name: 'canvas-invocation-preparation',
});
