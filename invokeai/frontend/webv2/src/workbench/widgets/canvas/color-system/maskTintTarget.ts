import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * Track the transient mask-tint editing target; clear it when it is no longer the selected mask. Tint itself
 * remains document state.
 */
export const maskTintTargetStore = createExternalStore<{ layerId: string | null }>({ layerId: null });

/**
 * Records the Color pane's tint gesture that previews but has not yet recorded. The canvas allows one structural
 * preview at a time, so an Alt-pick into the tint records this gesture before it starts its own; otherwise the
 * pane's late settle would land after the pick and overwrite it.
 */
let pendingTintSettle: (() => void) | null = null;

registerAccountOwnedResource({
  clear: () => {
    maskTintTargetStore.setSnapshot({ layerId: null });
    pendingTintSettle = null;
  },
  name: 'mask-tint-target',
});

/** Registers `settle` as the gesture an Alt-pick must record first. */
export const holdPendingMaskTint = (settle: () => void): void => {
  pendingTintSettle = settle;
};

/** Forgets `settle` once its gesture recorded, leaving a newer gesture's registration in place. */
export const releasePendingMaskTint = (settle: () => void): void => {
  if (pendingTintSettle === settle) {
    pendingTintSettle = null;
  }
};

/** Records any pending pane tint gesture now, before another tint preview begins. */
export const settlePendingMaskTint = (): void => {
  pendingTintSettle?.();
};

export const armMaskTintTarget = (layerId: string): void => {
  maskTintTargetStore.setSnapshot({ layerId });
};

export const clearMaskTintTarget = (): void => {
  if (maskTintTargetStore.getSnapshot().layerId !== null) {
    maskTintTargetStore.setSnapshot({ layerId: null });
  }
};

export const useMaskTintTargetLayerId = (): string | null =>
  maskTintTargetStore.useSelector((snapshot) => snapshot.layerId, Object.is);
