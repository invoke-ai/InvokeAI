import type { AppStore, RootState } from 'app/store/store';
import {
  isErrorChanged,
  isLoadingChanged,
  parsingErrorChanged,
  promptsChanged,
  randomSeedChanged,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getDynamicPromptsQueryArg } from 'features/dynamicPrompts/util/getDynamicPromptsQueryArg';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import { utilitiesApi } from 'services/api/endpoints/utilities';

const getRandomSeed = () => Date.now() + Math.floor(Math.random() * 1_000_000);

export const getShouldRefreshDynamicPromptsForEnqueue = (state: RootState): boolean => {
  const { dynamicPrompts } = state;

  if (dynamicPrompts.mode !== 'random' || dynamicPrompts.randomRefreshMode !== 'per_enqueue') {
    return false;
  }

  return getShouldProcessPrompt(selectPresetModifiedPrompts(state).positive);
};

/**
 * Random wildcard mode means "roll when I invoke". The preview remains useful, but the queue payload is refreshed here
 * so one-image-at-a-time workflows do not repeatedly enqueue the same cached random expansion.
 */
export const refreshDynamicPromptsForEnqueue = async (store: AppStore): Promise<RootState | null> => {
  const { dispatch, getState } = store;
  const state = getState();

  if (!getShouldRefreshDynamicPromptsForEnqueue(state)) {
    return state;
  }

  const prompt = selectPresetModifiedPrompts(state).positive;
  const { randomSamples, maxCombinations } = state.dynamicPrompts;
  const randomSeed = getRandomSeed();

  dispatch(randomSeedChanged(randomSeed));
  dispatch(isLoadingChanged(true));

  const queryArg = getDynamicPromptsQueryArg({
    prompt,
    mode: 'random',
    randomSamples,
    maxCombinations,
    randomSeed,
  });

  try {
    const req = dispatch(
      utilitiesApi.endpoints.dynamicPrompts.initiate(queryArg, { subscribe: false, forceRefetch: true })
    );
    const res = await req.unwrap();

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
