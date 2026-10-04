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
 * Acquire the active-submit acknowledgement before any async work; the Canvas orchestrator separately guards other
 * entry points.
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

  // After account invalidation, an old submission token must not release a new owner's lease for the same id.
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
