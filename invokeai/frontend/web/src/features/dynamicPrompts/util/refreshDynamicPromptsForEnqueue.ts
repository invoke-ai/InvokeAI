import type { AppStore, RootState } from 'app/store/store';
import {
  isErrorChanged,
  isLoadingChanged,
  parsingErrorChanged,
  promptsChanged,
  randomSeedChanged,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { resolveDynamicPrompts } from 'features/dynamicPrompts/util/resolveDynamicPrompts';
import { selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';

const getRandomSeed = () => Date.now() + Math.floor(Math.random() * 1_000_000);

export const getShouldRefreshDynamicPromptsForEnqueue = (state: RootState): boolean => {
  const { dynamicPrompts } = state;

  if (dynamicPrompts.mode !== 'random' || dynamicPrompts.randomRefreshMode === 'manual') {
    return false;
  }

  return getShouldProcessPrompt(selectPresetModifiedPrompts(state).positive);
};

/**
 * Dynamic prompts refresh before queueing unless the preview is locked. Per-image random mode and cycle-only per-invoke
 * mode ask for one resolved prompt per generated output, then the batch builder zips those prompts with seeds one-to-one.
 */
export const refreshDynamicPromptsForEnqueue = async (store: AppStore): Promise<RootState | null> => {
  const { dispatch, getState } = store;
  const state = getState();

  if (!getShouldRefreshDynamicPromptsForEnqueue(state)) {
    return state;
  }

  const prompt = selectPresetModifiedPrompts(state).positive;
  const { maxCombinations, randomRefreshMode, randomSamples } = state.dynamicPrompts;
  const randomSeed = getRandomSeed();

  dispatch(randomSeedChanged(randomSeed));
  dispatch(isLoadingChanged(true));

  try {
    const res = await resolveDynamicPrompts({
      dispatch,
      prompt,
      mode: 'random',
      randomSamples,
      maxCombinations,
      randomSeed,
      randomRefreshMode,
      iterations: state.params.iterations,
      forceRefetch: true,
    });

    dispatch(promptsChanged(res.prompts));
    dispatch(parsingErrorChanged(res.error));
    dispatch(isErrorChanged(false));

    return getState();
  } catch {
    dispatch(isErrorChanged(true));
    dispatch(isLoadingChanged(false));
    return null;
  }
};
