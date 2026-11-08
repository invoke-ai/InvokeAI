import { getGalleryBoardDestinationIds } from '@features/gallery/queries';
import { getGenerationDevicesSnapshot, resolveRandDeviceMetadata } from '@features/queue/devices';
import { getInvocationTemplatesSnapshot } from '@features/workflow/react';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { useExternalStoreSelector } from '@platform/state/selectors';

import type { InvocationRuntimeInputs, InvocationSourceId } from './invocationContracts';

interface InvocationPreparationSnapshot {
  leases: ReadonlyMap<string, number>;
}

export interface InvocationPreparationLease {
  projectId: string;
  token: number;
}

const EMPTY_LEASES: ReadonlyMap<string, number> = new Map();
const store = createExternalStoreCore<InvocationPreparationSnapshot>({ leases: EMPTY_LEASES });
let nextLeaseToken = 1;

/**
 * Single-flights a project's submission: acquire before the first await (lazy Canvas chunk, prompt expansion,
 * workflow generators) so a repeated invoke cannot submit the same captured settings twice. Canvas preparation
 * separately guards its own entry points.
 */
export const beginInvocationPreparation = (projectId: string): InvocationPreparationLease | null => {
  const { leases } = store.getSnapshot();

  if (leases.has(projectId)) {
    return null;
  }

  const lease = { projectId, token: nextLeaseToken };
  nextLeaseToken += 1;
  store.setSnapshot({ leases: new Map([...leases, [projectId, lease.token]]) });
  return lease;
};

export const endInvocationPreparation = (lease: InvocationPreparationLease): void => {
  const { leases } = store.getSnapshot();

  // After account invalidation, an old submission token must not release a new owner's lease for the same id.
  if (leases.get(lease.projectId) !== lease.token) {
    return;
  }

  const nextLeases = new Map(leases);
  nextLeases.delete(lease.projectId);
  store.setSnapshot({ leases: nextLeases.size > 0 ? nextLeases : EMPTY_LEASES });
};

export const isInvocationPreparing = (projectId: string): boolean => store.getSnapshot().leases.has(projectId);

export const useIsInvocationPreparing = (projectId: string): boolean =>
  useExternalStoreSelector(store.subscribe, store.getSnapshot, (snapshot) => snapshot.leases.has(projectId));

registerAccountOwnedResource({
  clear: () => store.setSnapshot({ leases: EMPTY_LEASES }),
  name: 'invocation-preparation',
});

/** Capture after async prerequisites; the template owner replaces snapshots, so its published value is safe to share. */
export const captureInvocationRuntimeInputs = (sourceId: InvocationSourceId): InvocationRuntimeInputs => ({
  noiseDevice: resolveRandDeviceMetadata(false, getGenerationDevicesSnapshot().options),
  templates:
    sourceId === 'workflow' ? getInvocationTemplatesSnapshot() : { error: null, status: 'idle', templates: {} },
  boardDestinationIds: getGalleryBoardDestinationIds(),
});
