import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

const store = createExternalStore<{ activeLayerIds: ReadonlySet<string> }>({ activeLayerIds: new Set<string>() });

registerAccountOwnedResource({
  clear: () => {
    store.setSnapshot({ activeLayerIds: new Set<string>() });
  },
  name: 'hotkey-modal-layers',
});

export const registerHotkeyModalLayer = (id: string): (() => void) => {
  store.patchSnapshot({ activeLayerIds: new Set([...store.getSnapshot().activeLayerIds, id]) });

  return () => {
    const activeLayerIds = new Set(store.getSnapshot().activeLayerIds);

    activeLayerIds.delete(id);
    store.patchSnapshot({ activeLayerIds });
  };
};

export const isHotkeyModalLayerActive = (): boolean => store.getSnapshot().activeLayerIds.size > 0;

export const useIsHotkeyModalLayerActive = (): boolean =>
  store.useSelector((snapshot) => snapshot.activeLayerIds.size > 0);
