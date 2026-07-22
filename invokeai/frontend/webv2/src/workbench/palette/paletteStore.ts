import { createExternalStore } from '@platform/state/externalStore';

/**
 * Open/close state for the command palette, addressable from anywhere: the
 * top-bar buttons, the `app.openCommandPalette` command, and the Launchpad's
 * own mod+K binding all funnel through here. The dialog is hosted per surface
 * (WorkbenchCommandPalette in the editor, LaunchpadCommandPalette on the
 * launchpad) and subscribes to this store.
 */

export const commandPaletteStore = createExternalStore<{ isOpen: boolean }>({ isOpen: false });

export const openCommandPalette = (): void => {
  commandPaletteStore.setSnapshot({ isOpen: true });
};

export const closeCommandPalette = (): void => {
  commandPaletteStore.setSnapshot({ isOpen: false });
};

export const toggleCommandPalette = (): void => {
  commandPaletteStore.setSnapshot({ isOpen: !commandPaletteStore.getSnapshot().isOpen });
};
