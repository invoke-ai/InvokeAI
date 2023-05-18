import { PostprocessingState } from './postprocessingSlice';

/**
 * Postprocessing slice persist denylist
 */
export const postprocessingPersistDenylist: (keyof PostprocessingState)[] = [];
