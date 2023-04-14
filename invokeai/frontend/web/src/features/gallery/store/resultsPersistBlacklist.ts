import { ResultsState } from './resultsSlice';

/**
 * Results slice persist blacklist
 *
 * Currently blacklisting results slice entirely, see persist config in store.ts
 */
const itemsToBlacklist: (keyof ResultsState)[] = [];

export const resultsBlacklist = itemsToBlacklist.map(
  (blacklistItem) => `results.${blacklistItem}`
);
