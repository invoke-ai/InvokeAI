import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { useSyncExternalStore } from 'react';

/**
 * Open/close state for the command palette, addressable from anywhere: the
 * top-bar buttons, the `app.openCommandPalette` command, and the Launchpad's
 * own mod+K binding all funnel through here. The dialog is hosted per surface
 * (WorkbenchCommandPalette in the editor, LaunchpadCommandPalette on the
 * launchpad) and subscribes to this store.
 */

export const commandPaletteStore = createExternalStoreCore<{ isOpen: boolean }>({ isOpen: false });

export const useIsCommandPaletteOpen = (): boolean =>
  useSyncExternalStore(commandPaletteStore.subscribe, commandPaletteStore.getSnapshot, commandPaletteStore.getSnapshot)
    .isOpen;

let returnFocusElement: HTMLElement | null = null;

const captureReturnFocusElement = (): void => {
  const activeElement = document.activeElement;

  returnFocusElement = activeElement instanceof HTMLElement ? activeElement : null;
};

export const getCommandPaletteReturnFocusElement = (): HTMLElement | null =>
  returnFocusElement?.isConnected ? returnFocusElement : null;

export const openCommandPalette = (): void => {
  if (!commandPaletteStore.getSnapshot().isOpen) {
    captureReturnFocusElement();
  }
  commandPaletteStore.setSnapshot({ isOpen: true });
};

export const closeCommandPalette = (): void => {
  commandPaletteStore.setSnapshot({ isOpen: false });
};

export const toggleCommandPalette = (): void => {
  if (commandPaletteStore.getSnapshot().isOpen) {
    closeCommandPalette();
  } else {
    openCommandPalette();
  }
};
