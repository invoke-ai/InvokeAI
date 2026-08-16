import type { GenerateLora, MainModelConfig } from '@features/generation/contracts';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';

import { describe, expect, it } from 'vitest';

import {
  areInputImagesEquivalent,
  areLorasEquivalent,
  areModelsEquivalent,
  areStringArraysEqual,
  getModelTriggerPhrases,
} from './upscaleComparators';

const model = (overrides: Partial<MainModelConfig> = {}): UpscaleWidgetValues['model'] =>
  ({ base: 'sdxl', key: 'model-a', name: 'Model A', trigger_phrases: ['alpha'], ...overrides }) as MainModelConfig;

const lora = (overrides: Partial<GenerateLora> = {}): GenerateLora =>
  ({ isEnabled: true, model: { key: 'lora-a' }, weight: 0.75, ...overrides }) as GenerateLora;

const image = (overrides: Partial<{ height: number; image_name: string; width: number }> = {}) =>
  ({ height: 512, image_name: 'a.png', width: 512, ...overrides }) as UpscaleWidgetValues['inputImage'];

describe('areStringArraysEqual', () => {
  it('compares by content and position, short-circuiting on identity', () => {
    const phrases = ['alpha', 'beta'];

    expect(areStringArraysEqual(phrases, phrases)).toBe(true);
    expect(areStringArraysEqual(['alpha', 'beta'], ['alpha', 'beta'])).toBe(true);
    expect(areStringArraysEqual(['alpha', 'beta'], ['beta', 'alpha'])).toBe(false);
    expect(areStringArraysEqual(['alpha'], ['alpha', 'beta'])).toBe(false);
  });
});

describe('getModelTriggerPhrases', () => {
  it('is empty without a model, and drops non-string entries a foreign config could carry', () => {
    expect(getModelTriggerPhrases(null)).toEqual([]);
    expect(getModelTriggerPhrases(model({ trigger_phrases: undefined }))).toEqual([]);
    expect(getModelTriggerPhrases(model({ trigger_phrases: ['alpha', 7, null] as unknown as string[] }))).toEqual([
      'alpha',
    ]);
  });
});

describe('areModelsEquivalent', () => {
  it('treats a fresh config under the same key as equivalent when its data matches', () => {
    // A catalog refresh replaces the object without changing anything the
    // prompt editors read, and re-rendering them for that is pure waste.
    expect(areModelsEquivalent(model(), model())).toBe(true);
  });

  it('separates models the prompt editors would render differently', () => {
    // Each of these feeds a prompt-editor affordance, so matching on `key`
    // alone would leave the editors showing the previous model's data.
    expect(areModelsEquivalent(model(), model({ key: 'model-b' }))).toBe(false);
    expect(areModelsEquivalent(model(), model({ base: 'sd-1' }))).toBe(false);
    expect(areModelsEquivalent(model(), model({ name: 'Model B' }))).toBe(false);
    expect(areModelsEquivalent(model(), model({ trigger_phrases: ['beta'] }))).toBe(false);
  });

  it('handles one side being absent', () => {
    expect(areModelsEquivalent(null, null)).toBe(true);
    expect(areModelsEquivalent(model(), null)).toBe(false);
    expect(areModelsEquivalent(null, model())).toBe(false);
  });
});

describe('areLorasEquivalent', () => {
  it('compares key, enablement, and weight per position', () => {
    expect(areLorasEquivalent([lora()], [lora()])).toBe(true);
    expect(areLorasEquivalent([lora()], [lora({ weight: 0.5 })])).toBe(false);
    expect(areLorasEquivalent([lora()], [lora({ isEnabled: false })])).toBe(false);
    expect(areLorasEquivalent([lora()], [lora({ model: { key: 'lora-b' } as GenerateLora['model'] })])).toBe(false);
  });

  it('separates lists of different length', () => {
    expect(areLorasEquivalent([lora()], [])).toBe(false);
    expect(areLorasEquivalent([], [lora()])).toBe(false);
    expect(areLorasEquivalent([], [])).toBe(true);
  });
});

describe('areInputImagesEquivalent', () => {
  it('separates images by dimensions as well as by name', () => {
    // The preflight readout is computed from these, so a same-named image at
    // new dimensions must not reuse the previous megapixel estimate.
    expect(areInputImagesEquivalent(image(), image())).toBe(true);
    expect(areInputImagesEquivalent(image(), image({ width: 1024 }))).toBe(false);
    expect(areInputImagesEquivalent(image(), image({ height: 1024 }))).toBe(false);
    expect(areInputImagesEquivalent(image(), image({ image_name: 'b.png' }))).toBe(false);
  });

  it('handles one side being absent', () => {
    expect(areInputImagesEquivalent(null, null)).toBe(true);
    expect(areInputImagesEquivalent(image(), null)).toBe(false);
    expect(areInputImagesEquivalent(null, image())).toBe(false);
  });
});
