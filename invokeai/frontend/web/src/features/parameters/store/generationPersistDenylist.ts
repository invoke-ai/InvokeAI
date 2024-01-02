import type { GenerationState } from './types';

/**
 * Generation slice persist denylist
 */
export const generationPersistDenylist: (keyof GenerationState)[] = [];
