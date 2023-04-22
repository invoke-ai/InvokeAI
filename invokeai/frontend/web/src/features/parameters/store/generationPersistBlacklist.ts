import { GenerationState } from './generationSlice';

/**
 * Generation slice persist blacklist
 */
const itemsToBlacklist: (keyof GenerationState)[] = [];

export const generationBlacklist = itemsToBlacklist.map(
  (blacklistItem) => `generation.${blacklistItem}`
);
