import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/** Open state for the Widgets dialog, so the palette, the app menu and the rails' menus open the one the shell hosts. */
const INITIAL_SNAPSHOT = { isOpen: false };

export const widgetsDialogStore = createExternalStore<{ isOpen: boolean }>(INITIAL_SNAPSHOT);

/**
 * Where focus goes when the dialog closes. Openers name a control that outlives them (the app menu's trigger, a rail's
 * menu trigger), since the menu item that opened the dialog is gone by then.
 */
let returnFocusTarget: (() => HTMLElement | null) | null = null;

registerAccountOwnedResource({
  clear: () => {
    returnFocusTarget = null;
    widgetsDialogStore.setSnapshot(INITIAL_SNAPSHOT);
  },
  name: 'widgets-dialog',
});

/** Opens the dialog; on close, focus goes back where the dialog found it (palette and shortcut openers). */
export const openWidgetsDialog = (): void => {
  returnFocusTarget = null;
  widgetsDialogStore.setSnapshot({ isOpen: true });
};

/** Opens the dialog from a menu item, naming the control focus returns to on close. */
export const openWidgetsDialogFrom = (returnFocus: () => HTMLElement | null): void => {
  returnFocusTarget = returnFocus;
  widgetsDialogStore.setSnapshot({ isOpen: true });
};

export const closeWidgetsDialog = (): void => widgetsDialogStore.setSnapshot(INITIAL_SNAPSHOT);

/** The control the opener named, while it is still on the page; otherwise the dialog's own restore applies. */
export const getWidgetsDialogReturnFocus = (): HTMLElement | null => {
  const target = returnFocusTarget?.() ?? null;

  return target?.isConnected ? target : null;
};
