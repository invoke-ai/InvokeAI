import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { useSyncExternalStore } from 'react';

/**
 * Share palette visibility across topbar, editor command, and Launchpad shortcut; each surface hosts its own
 * subscribed dialog.
 */

export const commandPaletteStore = createExternalStoreCore<{ isOpen: boolean }>({ isOpen: false });

export const useIsCommandPaletteOpen = (): boolean =>
  useSyncExternalStore(commandPaletteStore.subscribe, commandPaletteStore.getSnapshot, commandPaletteStore.getSnapshot)
    .isOpen;

let returnFocusElement: HTMLElement | null = null;

registerAccountOwnedResource({
  clear: () => {
    returnFocusElement = null;
    commandPaletteStore.setSnapshot({ isOpen: false });
  },
  name: 'command-palette',
});

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
