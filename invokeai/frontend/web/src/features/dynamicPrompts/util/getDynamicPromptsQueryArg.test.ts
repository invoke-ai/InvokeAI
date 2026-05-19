import { describe, expect, it } from 'vitest';

import { getDynamicPromptsQueryArg } from './getDynamicPromptsQueryArg';

describe('getDynamicPromptsQueryArg', () => {
  it('passes random mode with an explicit seed', () => {
    expect(
      getDynamicPromptsQueryArg({
        prompt: '__camera/lens__',
        mode: 'random',
        randomSamples: 1,
        maxCombinations: 100,
        randomSeed: 42,
      })
    ).toEqual({
      prompt: '__camera/lens__',
      max_prompts: 1,
      combinatorial: false,
      seed: 42,
    });
  });

  it('passes all-combinations mode explicitly', () => {
    expect(
      getDynamicPromptsQueryArg({
        prompt: '__camera/lens__',
        mode: 'combinatorial',
        randomSamples: 1,
        maxCombinations: 100,
        randomSeed: 42,
      })
    ).toEqual({
      prompt: '__camera/lens__',
      max_prompts: 100,
      combinatorial: true,
    });
  });
});
