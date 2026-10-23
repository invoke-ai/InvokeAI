import type { AdjustmentOwnerStatus, AdjustmentStatusEngine } from './AdjustmentStatusIcon';

/** A keyed status store standing in for the engine's, which tests drive with `set`. */
export const createFakeAdjustmentStatus = (): AdjustmentStatusEngine & {
  set(ownerId: string, status: AdjustmentOwnerStatus): void;
} => {
  const statuses = new Map<string, AdjustmentOwnerStatus>();
  const listeners = new Map<string, Set<() => void>>();
  return {
    interaction: {
      getLayerAdjustmentStatus: (ownerId) => statuses.get(ownerId) ?? 'idle',
      subscribeLayerAdjustmentStatus: (ownerId, listener) => {
        const keyed = listeners.get(ownerId) ?? new Set();
        listeners.set(ownerId, keyed);
        keyed.add(listener);
        return () => keyed.delete(listener);
      },
    },
    set: (ownerId, status) => {
      statuses.set(ownerId, status);
      listeners.get(ownerId)?.forEach((listener) => listener());
    },
  };
};
