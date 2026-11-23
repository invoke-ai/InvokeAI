import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/** Share workspace-manager visibility across triggers; the topbar hosts one dialog independently of menu mounting. */
interface LayoutPresetManagerSnapshot {
  /** A workspace change waiting for its confirmation: deleting a custom one, or resetting a built-in to its default. */
  confirm: { action: 'delete' | 'restore'; presetId: string } | null;
  isOpen: boolean;
  /** The tab whose menu opened the manager, where focus returns; kept past closing for the dialog's exit. */
  openerPresetId: string | null;
}

const INITIAL_SNAPSHOT: LayoutPresetManagerSnapshot = { confirm: null, isOpen: false, openerPresetId: null };

export const layoutPresetManagerStore = createExternalStore<LayoutPresetManagerSnapshot>(INITIAL_SNAPSHOT);

registerAccountOwnedResource({
  clear: () => {
    layoutPresetManagerStore.setSnapshot(INITIAL_SNAPSHOT);
  },
  name: 'layout-preset-manager',
});

export const openLayoutPresetManager = (openerPresetId: string | null = null): void =>
  layoutPresetManagerStore.setSnapshot({ ...layoutPresetManagerStore.getSnapshot(), isOpen: true, openerPresetId });

export const closeLayoutPresetManager = (): void =>
  layoutPresetManagerStore.setSnapshot({
    ...INITIAL_SNAPSHOT,
    openerPresetId: layoutPresetManagerStore.getSnapshot().openerPresetId,
  });

export const openLayoutPresetDelete = (presetId: string): void =>
  layoutPresetManagerStore.setSnapshot({
    ...layoutPresetManagerStore.getSnapshot(),
    confirm: { action: 'delete', presetId },
  });

export const openLayoutPresetRestore = (presetId: string): void =>
  layoutPresetManagerStore.setSnapshot({
    ...layoutPresetManagerStore.getSnapshot(),
    confirm: { action: 'restore', presetId },
  });

export const closeLayoutPresetAdmin = (): void =>
  layoutPresetManagerStore.setSnapshot({ ...layoutPresetManagerStore.getSnapshot(), confirm: null });
