import { ModelsState } from './modelSlice';

/**
 * Models slice persist denylist
 */
export const modelsPersistDenylist: (keyof ModelsState)[] = ['entities', 'ids'];
