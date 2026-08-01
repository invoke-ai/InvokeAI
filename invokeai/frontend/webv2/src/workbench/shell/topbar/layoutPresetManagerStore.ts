import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * Open/close state for the layout preset manager. Addressable from anywhere the
 * `Manage presets` entry appears; the dialog itself is hosted once by the top
 * bar so opening it never depends on which menu is currently mounted.
 */
interface LayoutPresetManagerSnapshot {
  deletePresetId: string | null;
  editPresetId: string | null;
  isOpen: boolean;
}

const INITIAL_SNAPSHOT: LayoutPresetManagerSnapshot = { deletePresetId: null, editPresetId: null, isOpen: false };

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

export const openLayoutPresetEdit = (presetId: string): void =>
  layoutPresetManagerStore.setSnapshot({
    ...layoutPresetManagerStore.getSnapshot(),
    deletePresetId: null,
    editPresetId: presetId,
  });

export const openLayoutPresetDelete = (presetId: string): void =>
  layoutPresetManagerStore.setSnapshot({
    ...layoutPresetManagerStore.getSnapshot(),
    deletePresetId: presetId,
    editPresetId: null,
  });

export const closeLayoutPresetAdmin = (): void =>
  layoutPresetManagerStore.setSnapshot({
    ...layoutPresetManagerStore.getSnapshot(),
    deletePresetId: null,
    editPresetId: null,
  });
