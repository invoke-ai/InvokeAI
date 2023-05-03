import { UIState } from './uiTypes';

/**
 * UI slice persist denylist
 */
const itemsToDenylist: (keyof UIState)[] = ['floatingProgressImageRect'];

export const uiDenylist = itemsToDenylist.map(
  (denylistItem) => `ui.${denylistItem}`
);
