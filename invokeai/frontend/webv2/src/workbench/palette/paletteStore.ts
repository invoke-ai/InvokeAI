import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { useSyncExternalStore } from 'react';

/** Palette visibility shared by topbar, editor command and Launchpad; each hosts its own dialog. */

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

export const getCommandPaletteReturnFocusElement = (): HTMLElement | null =>
  returnFocusElement?.isConnected ? returnFocusElement : null;

/** Closing focuses `returnFocus` (for a control gone by then), else what had focus at open. */
export const openCommandPaletteFrom = (returnFocus: HTMLElement | null): void => {
  if (!commandPaletteStore.getSnapshot().isOpen) {
    const target = returnFocus ?? document.activeElement;
    returnFocusElement = target instanceof HTMLElement ? target : null;
  }
  commandPaletteStore.setSnapshot({ isOpen: true });
};

export const openCommandPalette = (): void => openCommandPaletteFrom(null);

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
