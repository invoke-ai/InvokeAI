import { PostprocessingState } from './postprocessingSlice';

/**
 * Postprocessing slice persist blacklist
 */
const itemsToBlacklist: (keyof PostprocessingState)[] = [];

export const postprocessingBlacklist = itemsToBlacklist.map(
  (blacklistItem) => `postprocessing.${blacklistItem}`
);
