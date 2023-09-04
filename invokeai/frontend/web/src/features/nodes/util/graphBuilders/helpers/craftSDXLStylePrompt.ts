import { RootState } from 'app/store/store';

export const craftSDXLStylePrompt = (
  state: RootState,
  shouldConcatSDXLStylePrompt: boolean
) => {
  const { positivePrompt, negativePrompt } = state.generation;
  const { positiveStylePrompt, negativeStylePrompt } = state.sdxl;

  let craftedPositiveStylePrompt = positiveStylePrompt;
  let craftedNegativeStylePrompt = negativeStylePrompt;

  if (shouldConcatSDXLStylePrompt) {
    if (positiveStylePrompt.length > 0) {
      craftedPositiveStylePrompt = `${positivePrompt} ${positiveStylePrompt}`;
    } else {
      craftedPositiveStylePrompt = positivePrompt;
    }

    if (negativeStylePrompt.length > 0) {
      craftedNegativeStylePrompt = `${negativePrompt} ${negativeStylePrompt}`;
    } else {
      craftedNegativeStylePrompt = negativePrompt;
    }
  }

  return { craftedPositiveStylePrompt, craftedNegativeStylePrompt };
};
