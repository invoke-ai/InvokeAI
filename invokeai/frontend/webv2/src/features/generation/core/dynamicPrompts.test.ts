import { describe, expect, it } from 'vitest';

import type { GeneratePromptBatchPlanInput } from './dynamicPrompts';

import {
  buildGeneratePromptBatchPlan,
  hasDynamicPromptSyntax,
  sanitizeDynamicPromptsConfig,
  sanitizeMaxPrompts,
} from './dynamicPrompts';

const baseInput = (overrides: Partial<GeneratePromptBatchPlanInput> = {}): GeneratePromptBatchPlanInput => ({
  batchCount: 1,
  negativePrompt: 'blurry',
  negativePromptNodeId: 'negative_prompt',
  positivePromptNodeId: 'positive_prompt',
  prompts: ['a cat'],
  seed: 100,
  seedBehaviour: 'per-iteration',
  seedNodeId: 'seed',
  shouldRandomizeSeed: false,
  ...overrides,
});

describe('hasDynamicPromptSyntax', () => {
  it('detects a variant anywhere in the prompt', () => {
    expect(hasDynamicPromptSyntax('a {red|green} ball')).toBe(true);
    expect(hasDynamicPromptSyntax('{a}')).toBe(true);
    expect(hasDynamicPromptSyntax('multi\nline {a|b}')).toBe(true);
  });

  it('ignores prompts with no braces, including bare wildcards', () => {
    expect(hasDynamicPromptSyntax('a red ball')).toBe(false);
    expect(hasDynamicPromptSyntax('a __color__ ball')).toBe(false);
    expect(hasDynamicPromptSyntax('unclosed { brace')).toBe(false);
  });
});

describe('sanitizeMaxPrompts', () => {
  it('clamps to the backend bounds and falls back on garbage', () => {
    expect(sanitizeMaxPrompts(50)).toBe(50);
    expect(sanitizeMaxPrompts(0)).toBe(1);
    expect(sanitizeMaxPrompts(99_999)).toBe(10_000);
    expect(sanitizeMaxPrompts(12.6)).toBe(13);
    expect(sanitizeMaxPrompts('many')).toBe(100);
    expect(sanitizeMaxPrompts(undefined)).toBe(100);
  });
});

describe('sanitizeDynamicPromptsConfig', () => {
  it('returns null for non-object values', () => {
    expect(sanitizeDynamicPromptsConfig(null)).toBeNull();
    expect(sanitizeDynamicPromptsConfig('nope')).toBeNull();
  });

  it('defaults every unusable field', () => {
    expect(sanitizeDynamicPromptsConfig({})).toEqual({
      combinatorial: true,
      maxPrompts: 100,
      seedBehaviour: 'per-iteration',
    });
    expect(sanitizeDynamicPromptsConfig({ combinatorial: false, maxPrompts: 7, seedBehaviour: 'per-image' })).toEqual({
      combinatorial: false,
      maxPrompts: 7,
      seedBehaviour: 'per-image',
    });
    expect(sanitizeDynamicPromptsConfig({ seedBehaviour: 'PER_PROMPT' })?.seedBehaviour).toBe('per-iteration');
  });
});

describe('buildGeneratePromptBatchPlan with a single prompt', () => {
  // These two pin the pre-dynamic-prompts payload built by `enqueueGenerate`.
  it('matches the fixed-seed payload: one datum each, runs carries the batch count', () => {
    const plan = buildGeneratePromptBatchPlan(baseInput({ batchCount: 3, shouldRandomizeSeed: false }));

    expect(plan.data).toEqual([
      [
        { field_name: 'value', items: [100], node_path: 'seed' },
        { field_name: 'value', items: ['a cat'], node_path: 'positive_prompt' },
        { field_name: 'value', items: ['blurry'], node_path: 'negative_prompt' },
      ],
    ]);
    expect(plan.runs).toBe(3);
    expect(plan.expectedImageCount).toBe(3);
  });

  it('matches the randomized-seed payload: a seed sequence zipped with repeated prompts', () => {
    const plan = buildGeneratePromptBatchPlan(baseInput({ batchCount: 3, shouldRandomizeSeed: true }));

    expect(plan.data).toEqual([
      [
        { field_name: 'value', items: [100, 101, 102], node_path: 'seed' },
        { field_name: 'value', items: ['a cat', 'a cat', 'a cat'], node_path: 'positive_prompt' },
        { field_name: 'value', items: ['blurry', 'blurry', 'blurry'], node_path: 'negative_prompt' },
      ],
    ]);
    expect(plan.runs).toBe(1);
    expect(plan.expectedImageCount).toBe(3);
  });

  it('ignores the seed behaviour, which only has meaning across a prompt set', () => {
    const perIteration = buildGeneratePromptBatchPlan(baseInput({ batchCount: 2, seedBehaviour: 'per-iteration' }));
    const perImage = buildGeneratePromptBatchPlan(baseInput({ batchCount: 2, seedBehaviour: 'per-image' }));

    expect(perImage).toEqual(perIteration);
  });
});

describe('buildGeneratePromptBatchPlan with several prompts', () => {
  const prompts = ['a red cat', 'a green cat', 'a blue cat'];

  it('per-iteration keeps seeds in their own dimension so a seed spans the prompt set', () => {
    const plan = buildGeneratePromptBatchPlan(
      baseInput({ batchCount: 2, prompts, shouldRandomizeSeed: true, seedBehaviour: 'per-iteration' })
    );

    expect(plan.data).toEqual([
      [{ field_name: 'value', items: [100, 101], node_path: 'seed' }],
      [
        { field_name: 'value', items: prompts, node_path: 'positive_prompt' },
        { field_name: 'value', items: ['blurry', 'blurry', 'blurry'], node_path: 'negative_prompt' },
      ],
    ]);
    expect(plan.runs).toBe(1);
    expect(plan.expectedImageCount).toBe(6);
  });

  it('per-iteration with a fixed seed leans on runs for the iterations', () => {
    const plan = buildGeneratePromptBatchPlan(
      baseInput({ batchCount: 2, prompts, shouldRandomizeSeed: false, seedBehaviour: 'per-iteration' })
    );

    expect(plan.data[0]).toEqual([{ field_name: 'value', items: [100], node_path: 'seed' }]);
    expect(plan.runs).toBe(2);
    expect(plan.expectedImageCount).toBe(6);
  });

  it('per-image gives every generated image its own seed', () => {
    const plan = buildGeneratePromptBatchPlan(
      baseInput({ batchCount: 2, prompts, shouldRandomizeSeed: false, seedBehaviour: 'per-image' })
    );

    expect(plan.data).toEqual([
      [
        { field_name: 'value', items: [100, 101, 102, 103, 104, 105], node_path: 'seed' },
        { field_name: 'value', items: [...prompts, ...prompts], node_path: 'positive_prompt' },
        { field_name: 'value', items: Array.from({ length: 6 }, () => 'blurry'), node_path: 'negative_prompt' },
      ],
    ]);
    expect(plan.runs).toBe(1);
    expect(plan.expectedImageCount).toBe(6);
  });

  it('keeps every zipped group the same length, as the backend requires', () => {
    for (const seedBehaviour of ['per-iteration', 'per-image'] as const) {
      const plan = buildGeneratePromptBatchPlan(baseInput({ batchCount: 4, prompts, seedBehaviour }));

      for (const group of plan.data) {
        const lengths = new Set(group.map((datum) => datum.items.length));

        expect(lengths.size).toBe(1);
      }
    }
  });

  it('wraps seeds at the 32-bit ceiling', () => {
    const plan = buildGeneratePromptBatchPlan(
      baseInput({ batchCount: 1, prompts: ['a', 'b', 'c'], seed: 4_294_967_294, seedBehaviour: 'per-image' })
    );

    expect(plan.data[0][0].items).toEqual([4_294_967_294, 0, 1]);
  });

  it('falls back to an empty prompt rather than emitting an empty batch', () => {
    const plan = buildGeneratePromptBatchPlan(baseInput({ prompts: [] }));

    expect(plan.data[0][1].items).toEqual(['']);
    expect(plan.expectedImageCount).toBe(1);
  });
});
