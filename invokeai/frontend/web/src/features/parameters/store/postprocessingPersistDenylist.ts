import { PostprocessingState } from './postprocessingSlice';

/**
 * Postprocessing slice persist denylist
 */
const itemsToDenylist: (keyof PostprocessingState)[] = [];

export const postprocessingDenylist = itemsToDenylist.map(
  (denylistItem) => `postprocessing.${denylistItem}`
);
