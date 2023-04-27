import { GenerationState } from './generationSlice';

/**
 * Generation slice persist denylist
 */
const itemsToDenylist: (keyof GenerationState)[] = [];

export const generationDenylist = itemsToDenylist.map(
  (denylistItem) => `generation.${denylistItem}`
);
