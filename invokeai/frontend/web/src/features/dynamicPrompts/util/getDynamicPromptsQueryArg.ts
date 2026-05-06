import type { DynamicPromptMode } from 'features/dynamicPrompts/store/dynamicPromptsSlice';

type GetDynamicPromptsQueryArgArg = {
  prompt: string;
  mode: DynamicPromptMode;
  randomSamples: number;
  maxCombinations: number;
  randomSeed: number;
};

export const getDynamicPromptsQueryArg = ({
  prompt,
  mode,
  randomSamples,
  maxCombinations,
  randomSeed,
}: GetDynamicPromptsQueryArgArg) => {
  if (mode === 'combinatorial') {
    return {
      prompt,
      max_prompts: maxCombinations,
      combinatorial: true,
    };
  }

  return {
    prompt,
    max_prompts: randomSamples,
    combinatorial: false,
    seed: randomSeed,
  };
};
