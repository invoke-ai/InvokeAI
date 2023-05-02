import { ResultsState } from './resultsSlice';

/**
 * Results slice persist denylist
 *
 * Currently denylisting results slice entirely, see persist config in store.ts
 */
const itemsToDenylist: (keyof ResultsState)[] = [];

export const resultsDenylist = itemsToDenylist.map(
  (denylistItem) => `results.${denylistItem}`
);
