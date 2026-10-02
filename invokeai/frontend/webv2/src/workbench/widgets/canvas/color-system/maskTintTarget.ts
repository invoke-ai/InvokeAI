import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * Track the transient mask-tint editing target; clear it when it is no longer the selected mask. Tint itself
 * remains document state.
 */
export const maskTintTargetStore = createExternalStore<{ layerId: string | null }>({ layerId: null });

registerAccountOwnedResource({
  clear: () => {
    maskTintTargetStore.setSnapshot({ layerId: null });
  },
  name: 'mask-tint-target',
});

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
