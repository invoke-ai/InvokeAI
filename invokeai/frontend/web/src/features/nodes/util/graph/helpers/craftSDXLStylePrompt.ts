import type { RootState } from 'app/store/store';

export const buildSDXLStylePrompts = (
  state: RootState,
  overrideConcat?: boolean
) => {
  const { positivePrompt, negativePrompt } = state.generation;
  const {
    positiveStylePrompt,
    negativeStylePrompt,
    shouldConcatSDXLStylePrompt,
  } = state.sdxl;

  // Construct Style Prompt
  const joinedPositiveStylePrompt =
    shouldConcatSDXLStylePrompt || overrideConcat
      ? [positivePrompt, positiveStylePrompt].join(' ')
      : positiveStylePrompt;

  const joinedNegativeStylePrompt =
    shouldConcatSDXLStylePrompt || overrideConcat
      ? [negativePrompt, negativeStylePrompt].join(' ')
      : negativeStylePrompt;

  return { joinedPositiveStylePrompt, joinedNegativeStylePrompt };
};
