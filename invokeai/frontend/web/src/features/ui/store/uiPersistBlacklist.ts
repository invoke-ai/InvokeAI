import { UIState } from './uiTypes';

/**
 * UI slice persist blacklist
 */
const itemsToBlacklist: (keyof UIState)[] = [];

export const uiBlacklist = itemsToBlacklist.map(
  (blacklistItem) => `ui.${blacklistItem}`
);
