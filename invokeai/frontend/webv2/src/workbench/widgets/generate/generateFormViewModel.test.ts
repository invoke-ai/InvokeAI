import type { GenerateSettings } from '@workbench/generation/types';

import { describe, expect, it } from 'vitest';

import {
  areGenerateFormValuesEqual,
  areProjectGenerateFormValuesEqual,
  getGenerateFormCommitPatch,
  getSettingsWithLatestPromptFields,
} from './generateFormViewModel';

const createSettings = (overrides: Partial<GenerateSettings> = {}): GenerateSettings =>
  ({
    aspectRatioId: '1:1',
    aspectRatioIsLocked: false,
    aspectRatioValue: 1,
    batchCount: 1,
    cfgRescaleMultiplier: 0,
    cfgScale: 7,
    clipEmbedModel: null,
    clipGEmbedModel: null,
    clipLEmbedModel: null,
    clipSkip: 0,
    colorCompensation: false,
    componentSourceModel: null,
    height: 1024,
    loras: [],
    modelKey: 'model-a',
    negativePrompt: 'old negative',
    negativePromptEnabled: true,
    negativePromptHeightPx: 56,
    positivePrompt: 'old positive',
    positivePromptHeightPx: 96,
    qwen3EncoderModel: null,
    qwenVLEncoderModel: null,
    scheduler: 'euler_a',
    seamlessXAxis: false,
    seamlessYAxis: false,
    seed: 123,
    shouldRandomizeSeed: false,
    steps: 30,
    t5EncoderModel: null,
    vae: null,
    vaePrecision: 'fp32',
    width: 1024,
    ...overrides,
  }) as GenerateSettings;

describe('getSettingsWithLatestPromptFields', () => {
  it('preserves latest prompt fields when committing a full settings object', () => {
    const staleFullSettings = createSettings({ cfgScale: 9, modelKey: 'model-b', positivePrompt: 'stale positive' });
    const latestSettings = createSettings({
      negativePrompt: 'latest negative',
      negativePromptEnabled: false,
      negativePromptHeightPx: 72,
      positivePrompt: 'latest positive',
      positivePromptHeightPx: 144,
    });

    expect(getSettingsWithLatestPromptFields(staleFullSettings, latestSettings)).toMatchObject({
      cfgScale: 9,
      modelKey: 'model-b',
      negativePrompt: 'latest negative',
      negativePromptEnabled: false,
      negativePromptHeightPx: 72,
      positivePrompt: 'latest positive',
      positivePromptHeightPx: 144,
    });
  });
});

describe('areGenerateFormValuesEqual', () => {
  it('ignores values owned by prompt fields and topbar batch count', () => {
    expect(
      areGenerateFormValuesEqual(
        { batchCount: 1, positivePrompt: 'first', steps: 30 },
        { batchCount: 4, positivePrompt: 'second', steps: 30 }
      )
    ).toBe(true);
  });

  it('detects changes rendered by the form', () => {
    expect(areGenerateFormValuesEqual({ batchCount: 1, steps: 30 }, { batchCount: 1, steps: 40 })).toBe(false);
  });
});

describe('areProjectGenerateFormValuesEqual', () => {
  it('treats a project switch as a value change even when only ignored form fields differ', () => {
    expect(
      areProjectGenerateFormValuesEqual(
        { projectId: 'project-a', values: { positivePrompt: 'project a', steps: 30 } },
        { projectId: 'project-b', values: { positivePrompt: 'project b', steps: 30 } }
      )
    ).toBe(false);
  });

  it('still ignores prompt-owned and topbar-owned values within the same project', () => {
    expect(
      areProjectGenerateFormValuesEqual(
        { projectId: 'project-a', values: { batchCount: 1, positivePrompt: 'first', steps: 30 } },
        { projectId: 'project-a', values: { batchCount: 4, positivePrompt: 'second', steps: 30 } }
      )
    ).toBe(true);
  });
});

describe('getGenerateFormCommitPatch', () => {
  it('omits topbar-owned batch count from full form commits', () => {
    const patch = getGenerateFormCommitPatch(createSettings({ batchCount: 8, positivePrompt: 'keep prompt' }));

    expect(patch.batchCount).toBeUndefined();
    expect(patch.positivePrompt).toBe('keep prompt');
  });
});
