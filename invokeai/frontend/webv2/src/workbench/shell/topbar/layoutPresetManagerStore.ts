import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/** Share workspace-manager visibility across triggers; the topbar hosts one dialog independently of menu mounting. */
interface LayoutPresetManagerSnapshot {
  deletePresetId: string | null;
  isOpen: boolean;
}

const INITIAL_SNAPSHOT: LayoutPresetManagerSnapshot = { deletePresetId: null, isOpen: false };

export const layoutPresetManagerStore = createExternalStore<LayoutPresetManagerSnapshot>(INITIAL_SNAPSHOT);

registerAccountOwnedResource({
  clear: () => {
    layoutPresetManagerStore.setSnapshot(INITIAL_SNAPSHOT);
  },
  name: 'layout-preset-manager',
});

export const openLayoutPresetManager = (): void =>
  layoutPresetManagerStore.setSnapshot({ ...layoutPresetManagerStore.getSnapshot(), isOpen: true });

export const closeLayoutPresetManager = (): void => layoutPresetManagerStore.setSnapshot(INITIAL_SNAPSHOT);

export const openLayoutPresetDelete = (presetId: string): void =>
  layoutPresetManagerStore.setSnapshot({ ...layoutPresetManagerStore.getSnapshot(), deletePresetId: presetId });

export const closeLayoutPresetAdmin = (): void =>
  layoutPresetManagerStore.setSnapshot({ ...layoutPresetManagerStore.getSnapshot(), deletePresetId: null });
