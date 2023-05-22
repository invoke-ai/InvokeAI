import { ResultsState } from './resultsSlice';

/**
 * Results slice persist denylist
 *
 * Currently denylisting results slice entirely, see `serialize.ts`
 */
export const resultsPersistDenylist: (keyof ResultsState)[] = [];
