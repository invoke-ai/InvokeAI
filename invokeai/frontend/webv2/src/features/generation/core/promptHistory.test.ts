import { describe, expect, it } from 'vitest';

import type { MainModelConfig } from './types';

import { getDefaultGenerateSettings, getPromptHistoryRecallPatch } from './baseGenerationPolicies';

const createModel = (base: string): MainModelConfig => ({
  base,
  key: `${base}-model`,
  name: `${base} model`,
  type: 'main',
});

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
