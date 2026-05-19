import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { debounce } from 'es-toolkit/compat';
import { selectIterations } from 'features/controlLayers/store/paramsSlice';
import {
  type DynamicPromptMode,
  isErrorChanged,
  isLoadingChanged,
  parsingErrorChanged,
  promptsChanged,
  selectDynamicPromptsMaxCombinations,
  selectDynamicPromptsMode,
  selectDynamicPromptsRandomRefreshMode,
  selectDynamicPromptsRandomSamples,
  selectDynamicPromptsRandomSeed,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { resolveDynamicPrompts } from 'features/dynamicPrompts/util/resolveDynamicPrompts';
import { selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import { useEffect, useMemo } from 'react';

const DYNAMIC_PROMPTS_DEBOUNCE_MS = 1000;

/**
 * This hook watches for changes to state that should trigger dynamic prompts to be updated.
 */
export const useDynamicPromptsWatcher = () => {
  const { getState, dispatch } = useAppStore();
  // The prompt to process is derived from the preset-modified prompts
  const presetModifiedPrompts = useAppSelector(selectPresetModifiedPrompts);
  const mode = useAppSelector(selectDynamicPromptsMode);
  const randomSamples = useAppSelector(selectDynamicPromptsRandomSamples);
  const randomRefreshMode = useAppSelector(selectDynamicPromptsRandomRefreshMode);
  const maxCombinations = useAppSelector(selectDynamicPromptsMaxCombinations);
  const randomSeed = useAppSelector(selectDynamicPromptsRandomSeed);
  const iterations = useAppSelector(selectIterations);

  const debouncedUpdateDynamicPrompts = useMemo(
    () =>
      debounce(
        async (
          positivePrompt: string,
          mode: DynamicPromptMode,
          randomSamples: number,
          maxCombinations: number,
          randomSeed: number,
          randomRefreshMode: 'manual' | 'per_enqueue' | 'per_image',
          iterations: number
        ) => {
          // Try to fetch the dynamic prompts and store in state
          try {
            const res = await resolveDynamicPrompts({
              dispatch,
              prompt: positivePrompt,
              mode,
              randomSamples,
              maxCombinations,
              randomSeed,
              randomRefreshMode,
              iterations,
            });

            dispatch(promptsChanged(res.prompts));
            dispatch(parsingErrorChanged(res.error));
            dispatch(isErrorChanged(false));
          } catch {
            dispatch(isErrorChanged(true));
            dispatch(isLoadingChanged(false));
          }
        },
        DYNAMIC_PROMPTS_DEBOUNCE_MS
      ),
    [dispatch]
  );

  useEffect(() => {
    const state = getState();

    // If the prompt is not in the cache, check if we should process it - this is just looking for dynamic prompts syntax
    if (!getShouldProcessPrompt(presetModifiedPrompts.positive)) {
      dispatch(promptsChanged([presetModifiedPrompts.positive]));
      dispatch(parsingErrorChanged(undefined));
      dispatch(isErrorChanged(false));
      return;
    }

    // If we are here, we need to process the prompt
    if (!state.dynamicPrompts.isLoading) {
      dispatch(isLoadingChanged(true));
    }

    debouncedUpdateDynamicPrompts(
      presetModifiedPrompts.positive,
      mode,
      randomSamples,
      maxCombinations,
      randomSeed,
      randomRefreshMode,
      iterations
    );
  }, [
    debouncedUpdateDynamicPrompts,
    dispatch,
    getState,
    iterations,
    maxCombinations,
    mode,
    presetModifiedPrompts,
    randomSamples,
    randomRefreshMode,
    randomSeed,
  ]);
};
