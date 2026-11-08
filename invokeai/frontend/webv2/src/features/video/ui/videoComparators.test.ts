import type { MainModelConfig } from '@features/generation/contracts';

import { createDefaultVideoWidgetValues } from '@features/video/core/widgetValues';
import { describe, expect, it } from 'vitest';

import { areVideoValuesEqual } from './videoComparators';

/** Missing comparator fields suppress edits; cover each independently editable value. */
describe('areVideoValuesEqual', () => {
  const fl2vaBase: MainModelConfig = {
    base: 'minimax-h3',
    format: 'checkpoint',
    key: 'h3-fl2va-base',
    name: 'MiniMax H3 FL2VA Transformer',
    type: 'main',
    variant: 'fl2va',
  };
  const values = createDefaultVideoWidgetValues();

  it('treats the hybrid quality base and its start block as part of the values', () => {
    expect(areVideoValuesEqual(values, { ...values })).toBe(true);
    expect(areVideoValuesEqual(values, { ...values, h3HybridBaseModel: fl2vaBase })).toBe(false);
    expect(areVideoValuesEqual(values, { ...values, h3HybridStartBlock: values.h3HybridStartBlock + 1 })).toBe(false);
    expect(
      areVideoValuesEqual(
        { ...values, h3HybridBaseModel: fl2vaBase },
        { ...values, h3HybridBaseModel: { ...fl2vaBase } }
      )
    ).toBe(true);
  });

  it('treats the duration head and the auto duration switch as part of the values', () => {
    const head = { base: 'ltx-2', key: 'head', name: 'Duration head', type: 'ltx2_duration_head' } as never;

    expect(areVideoValuesEqual(values, { ...values, ltx2DurationHeadModel: head })).toBe(false);
    expect(areVideoValuesEqual(values, { ...values, autoDuration: !values.autoDuration })).toBe(false);
  });

  it('treats every dynamic prompts setting as part of the values', () => {
    const edits: Partial<typeof values>[] = [
      { dynamicPromptsCombinatorial: !values.dynamicPromptsCombinatorial },
      { dynamicPromptsEnabled: !values.dynamicPromptsEnabled },
      { dynamicPromptsMaxPrompts: values.dynamicPromptsMaxPrompts + 1 },
      { dynamicPromptsResample: !values.dynamicPromptsResample },
      { dynamicPromptsSampleSeed: values.dynamicPromptsSampleSeed + 1 },
      { dynamicPromptsSeedBehaviour: 'per-image' },
    ];

    for (const edit of edits) {
      expect(areVideoValuesEqual(values, { ...values, ...edit })).toBe(false);
    }
  });
});
