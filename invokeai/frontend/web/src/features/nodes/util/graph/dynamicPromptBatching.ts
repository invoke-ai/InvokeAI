import type { RootState } from 'app/store/store';
import { range } from 'es-toolkit/compat';
import type { SeedBehaviour } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { getHasCyclicWildcardSyntax } from 'features/dynamicPrompts/util/promptIntent';
import { selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';

export const getExtendedDynamicPrompts = ({
  seedBehaviour,
  iterations,
  prompts,
}: {
  seedBehaviour: SeedBehaviour;
  iterations: number;
  prompts: string[];
}): string[] => {
  if (seedBehaviour === 'PER_PROMPT') {
    return range(iterations).flatMap(() => prompts);
  }
  return prompts;
};

export const getShouldUsePerOutputDynamicPrompts = (state: RootState): boolean => {
  const prompt = selectPresetModifiedPrompts(state).positive;
  const { mode, randomRefreshMode } = state.dynamicPrompts;

  if (mode !== 'random' || !getShouldProcessPrompt(prompt)) {
    return false;
  }

  return randomRefreshMode === 'per_image' || getHasCyclicWildcardSyntax(prompt);
};
