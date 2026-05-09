import type { AppDispatch } from 'app/store/store';
import type {
  DynamicPromptMode,
  DynamicPromptRandomRefreshMode,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getDynamicPromptsQueryArg } from 'features/dynamicPrompts/util/getDynamicPromptsQueryArg';
import {
  getHasCyclicWildcardSyntax,
  getHasMixedCyclicAndNonCyclicDynamicPromptSyntax,
} from 'features/dynamicPrompts/util/promptIntent';
import { utilitiesApi } from 'services/api/endpoints/utilities';
import type { paths } from 'services/api/schema';

type DynamicPromptsResponse =
  paths['/api/v1/utilities/dynamicprompts']['post']['responses']['200']['content']['application/json'];

type ResolveDynamicPromptsArg = {
  dispatch: AppDispatch;
  prompt: string;
  mode: DynamicPromptMode;
  randomSamples: number;
  maxCombinations: number;
  randomSeed: number;
  randomRefreshMode: DynamicPromptRandomRefreshMode;
  iterations: number;
  forceRefetch?: boolean;
};

const CYCLIC_WILDCARD_PATTERN = /__@[^\r\n_][^\r\n]*?__/g;
const MAX_DYNAMIC_PROMPTS = 10000;

export const getDynamicPromptsOutputCount = ({
  iterations,
  prompt,
  randomRefreshMode,
  randomSamples,
}: {
  iterations: number;
  prompt: string;
  randomRefreshMode: DynamicPromptRandomRefreshMode;
  randomSamples: number;
}): number => {
  if (randomRefreshMode === 'per_image' || getHasCyclicWildcardSyntax(prompt)) {
    return randomSamples * Math.max(iterations, 1);
  }

  return randomSamples;
};

export const resolveDynamicPrompts = async ({
  dispatch,
  prompt,
  mode,
  randomSamples,
  maxCombinations,
  randomSeed,
  randomRefreshMode,
  iterations,
  forceRefetch = false,
}: ResolveDynamicPromptsArg): Promise<DynamicPromptsResponse> => {
  if (mode !== 'random') {
    return fetchDynamicPrompts({
      dispatch,
      prompt,
      mode,
      randomSamples,
      maxCombinations,
      randomSeed,
      forceRefetch,
    });
  }

  const outputCount = Math.min(
    getDynamicPromptsOutputCount({ iterations, prompt, randomRefreshMode, randomSamples }),
    MAX_DYNAMIC_PROMPTS
  );

  if (randomRefreshMode === 'per_image' || !getHasMixedCyclicAndNonCyclicDynamicPromptSyntax(prompt)) {
    return fetchDynamicPrompts({
      dispatch,
      prompt,
      mode,
      randomSamples: outputCount,
      maxCombinations,
      randomSeed,
      forceRefetch,
    });
  }

  const protectedPrompt = protectCyclicWildcards(prompt);
  const randomResponse = await fetchDynamicPrompts({
    dispatch,
    prompt: protectedPrompt.prompt,
    mode,
    randomSamples,
    maxCombinations,
    randomSeed,
    forceRefetch,
  });

  if (randomResponse.error || randomResponse.prompts.length === 0) {
    return randomResponse;
  }

  const cyclePromptCount = Math.max(1, Math.floor(outputCount / randomResponse.prompts.length));
  const cycleResponses = await Promise.all(
    randomResponse.prompts.map((randomPrompt) =>
      fetchDynamicPrompts({
        dispatch,
        prompt: restoreCyclicWildcards(randomPrompt, protectedPrompt.tokens),
        mode,
        randomSamples: cyclePromptCount,
        maxCombinations,
        randomSeed,
        forceRefetch,
      })
    )
  );

  return mergeDynamicPromptResponses(
    [randomResponse, ...cycleResponses],
    cycleResponses.flatMap((res) => res.prompts)
  );
};

const fetchDynamicPrompts = ({
  dispatch,
  prompt,
  mode,
  randomSamples,
  maxCombinations,
  randomSeed,
  forceRefetch,
}: {
  dispatch: AppDispatch;
  prompt: string;
  mode: DynamicPromptMode;
  randomSamples: number;
  maxCombinations: number;
  randomSeed: number;
  forceRefetch: boolean;
}): Promise<DynamicPromptsResponse> => {
  const queryArg = getDynamicPromptsQueryArg({
    prompt,
    mode,
    randomSamples,
    maxCombinations,
    randomSeed,
  });
  const req = dispatch(utilitiesApi.endpoints.dynamicPrompts.initiate(queryArg, { subscribe: false, forceRefetch }));
  return req.unwrap();
};

const protectCyclicWildcards = (prompt: string): { prompt: string; tokens: string[] } => {
  const tokens: string[] = [];
  const protectedPrompt = prompt.replace(CYCLIC_WILDCARD_PATTERN, (token) => {
    const index = tokens.push(token) - 1;
    return getCyclePlaceholder(index);
  });

  return { prompt: protectedPrompt, tokens };
};

const restoreCyclicWildcards = (prompt: string, tokens: string[]): string =>
  tokens.reduce((nextPrompt, token, index) => nextPrompt.replaceAll(getCyclePlaceholder(index), token), prompt);

const getCyclePlaceholder = (index: number): string => `<<INVOKE_CYCLE_${index}>>`;

const mergeDynamicPromptResponses = (
  responses: DynamicPromptsResponse[],
  prompts: string[]
): DynamicPromptsResponse => ({
  prompts: prompts.length > 0 ? prompts : [''],
  error: responses.map((res) => res.error).find(Boolean),
  warnings: [...new Set(responses.flatMap((res) => res.warnings ?? []))],
  missing_wildcards: [...new Set(responses.flatMap((res) => res.missing_wildcards ?? []))],
});
