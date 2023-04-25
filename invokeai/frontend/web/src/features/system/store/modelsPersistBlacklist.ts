import { ModelsState } from './modelSlice';

/**
 * Models slice persist blacklist
 */
const itemsToBlacklist: (keyof ModelsState)[] = ['entities', 'ids'];

export const modelsBlacklist = itemsToBlacklist.map(
  (blacklistItem) => `models.${blacklistItem}`
);
