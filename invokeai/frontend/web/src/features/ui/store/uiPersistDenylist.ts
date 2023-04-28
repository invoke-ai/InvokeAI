import { UIState } from './uiTypes';

/**
 * UI slice persist denylist
 */
const itemsToDenylist: (keyof UIState)[] = [];

export const uiDenylist = itemsToDenylist.map(
  (denylistItem) => `ui.${denylistItem}`
);
