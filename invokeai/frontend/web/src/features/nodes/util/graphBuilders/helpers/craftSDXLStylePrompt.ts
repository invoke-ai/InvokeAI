import { RootState } from 'app/store/store';

export const buildSDXLStylePrompts = (state: RootState) => {
  const { positivePrompt, negativePrompt } = state.generation;
  const {
    positiveStylePrompt,
    negativeStylePrompt,
    shouldConcatSDXLStylePrompt,
  } = state.sdxl;

  // Construct Style Prompt
  const joinedPositiveStylePrompt = shouldConcatSDXLStylePrompt
    ? [positivePrompt, positiveStylePrompt].join(' ')
    : positiveStylePrompt;

  const joinedNegativeStylePrompt = shouldConcatSDXLStylePrompt
    ? [negativePrompt, negativeStylePrompt].join(' ')
    : negativeStylePrompt;

  return { joinedPositiveStylePrompt, joinedNegativeStylePrompt };
};
