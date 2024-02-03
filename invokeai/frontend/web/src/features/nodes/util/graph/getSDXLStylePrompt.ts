import type { RootState } from 'app/store/store';

export const getSDXLStylePrompts = (state: RootState): { positiveStylePrompt: string; negativeStylePrompt: string } => {
  const { positivePrompt, negativePrompt } = state.generation;
  const { positiveStylePrompt, negativeStylePrompt, shouldConcatSDXLStylePrompt } = state.sdxl;

  return {
    positiveStylePrompt: shouldConcatSDXLStylePrompt ? positivePrompt : positiveStylePrompt,
    negativeStylePrompt: shouldConcatSDXLStylePrompt ? negativePrompt : negativeStylePrompt,
  };
};
