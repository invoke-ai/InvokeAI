import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import type { MainModelConfig } from './types';

import { getDefaultGenerateSettings, getPromptHistoryRecallPatch } from './baseGenerationPolicies';
import { MAX_PROMPT_HISTORY, restorePromptHistoryItems } from './promptHistory';

const createModel = (base: string): MainModelConfig => ({
  base,
  key: `${base}-model`,
  name: `${base} model`,
  type: 'main',
});

seedArchitectureCapabilities();

describe('getPromptHistoryRecallPatch', () => {
  it('clears an absent negative prompt when the selected model exposes it', () => {
    const model = createModel('sdxl');
    const values = { ...getDefaultGenerateSettings(model), negativePrompt: 'stale negative prompt' };

    expect(
      getPromptHistoryRecallPatch({
        item: { negativePrompt: null, positivePrompt: 'recalled prompt' },
        models: [model],
        values,
      })
    ).toEqual({ negativePrompt: '', negativePromptEnabled: true, positivePrompt: 'recalled prompt' });
  });

  it('does not mutate hidden negative-prompt state for models that do not support it', () => {
    const model = createModel('flux');
    const values = { ...getDefaultGenerateSettings(model), negativePrompt: 'keep hidden value' };

    expect(
      getPromptHistoryRecallPatch({
        item: { negativePrompt: 'history negative', positivePrompt: 'recalled prompt' },
        models: [model],
        values,
      })
    ).toEqual({ positivePrompt: 'recalled prompt' });
  });
});

describe('restorePromptHistoryItems', () => {
  const item = (positivePrompt: string) => ({ negativePrompt: null, positivePrompt });

  it('puts cleared entries back behind those added since, without listing one twice', () => {
    expect(
      restorePromptHistoryItems([item('newer'), item('b')], [item('a'), item('b'), item('c')]).map(
        (entry) => entry.positivePrompt
      )
    ).toEqual(['newer', 'b', 'a', 'c']);
  });

  it('keeps the history within its cap', () => {
    const cleared = Array.from({ length: MAX_PROMPT_HISTORY }, (_, index) => item(`old ${index}`));

    const restored = restorePromptHistoryItems([item('newer')], cleared);

    expect(restored).toHaveLength(MAX_PROMPT_HISTORY);
    expect(restored[0]?.positivePrompt).toBe('newer');
    expect(restored.at(-1)?.positivePrompt).toBe(`old ${MAX_PROMPT_HISTORY - 2}`);
  });
});
