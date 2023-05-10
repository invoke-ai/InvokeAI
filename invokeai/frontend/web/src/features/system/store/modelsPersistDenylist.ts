import { ModelsState } from './modelSlice';

/**
 * Models slice persist denylist
 */
const itemsToDenylist: (keyof ModelsState)[] = ['entities', 'ids'];

export const modelsDenylist = itemsToDenylist.map(
  (denylistItem) => `models.${denylistItem}`
);
