import { describe, expect, it } from 'vitest';

import type { GeneratePromptBatchPlanInput } from './promptBatch';

import {
  buildGeneratePromptBatchPlan,
  buildQueueWorkflowBatchPlan,
  generateSeedSequence,
  isQueueWorkflowBatchDatum,
} from './promptBatch';

const SEED_MAX = 4_294_967_295;

const baseInput = (overrides: Partial<GeneratePromptBatchPlanInput> = {}): GeneratePromptBatchPlanInput => ({
  batchCount: 1,
  negativePrompt: 'blurry',
  negativePromptNodeId: 'negative_prompt',
  positivePromptNodeId: 'positive_prompt',
  prompts: ['a cat'],
  seed: 100,
  seedBehaviour: 'per-iteration',
  seedNodeId: 'seed',
  seedStep: 0,
  ...overrides,
});

describe('generateSeedSequence', () => {
  it('wraps over the inclusive seed range in either direction', () => {
    expect(generateSeedSequence(SEED_MAX - 1, 3)).toEqual([SEED_MAX - 1, SEED_MAX, 0]);
    expect(generateSeedSequence(1, 3, -1)).toEqual([1, 0, SEED_MAX]);
  });
});

describe('buildGeneratePromptBatchPlan with a single prompt', () => {
  // These two pin the pre-dynamic-prompts payload built by `enqueueGenerate`.
  it('matches the held-seed payload: one datum each, runs carries the batch count', () => {
    const plan = buildGeneratePromptBatchPlan(baseInput({ batchCount: 3, seedStep: 0 }));

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

  it('matches the stepping-seed payload: a seed sequence zipped with repeated prompts', () => {
    const plan = buildGeneratePromptBatchPlan(baseInput({ batchCount: 3, seedStep: 1 }));

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

  it('counts down for a decrementing seed', () => {
    const plan = buildGeneratePromptBatchPlan(baseInput({ batchCount: 3, seedStep: -1 }));

    expect(plan.data[0][0].items).toEqual([100, 99, 98]);
    expect(plan.runs).toBe(1);
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
      baseInput({ batchCount: 2, prompts, seedBehaviour: 'per-iteration', seedStep: 1 })
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

  it('per-iteration with a held seed leans on runs for the iterations', () => {
    const plan = buildGeneratePromptBatchPlan(
      baseInput({ batchCount: 2, prompts, seedBehaviour: 'per-iteration', seedStep: 0 })
    );

    expect(plan.data[0]).toEqual([{ field_name: 'value', items: [100], node_path: 'seed' }]);
    expect(plan.runs).toBe(2);
    expect(plan.expectedImageCount).toBe(6);
  });

  it('per-image gives every generated image its own seed while the seed steps', () => {
    const plan = buildGeneratePromptBatchPlan(
      baseInput({ batchCount: 2, prompts, seedBehaviour: 'per-image', seedStep: 1 })
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

  it('per-image with a held seed still uses that one seed for every image', () => {
    const plan = buildGeneratePromptBatchPlan(
      baseInput({ batchCount: 2, prompts, seedBehaviour: 'per-image', seedStep: 0 })
    );

    expect(plan.data[0]).toEqual([{ field_name: 'value', items: [100], node_path: 'seed' }]);
    expect(plan.runs).toBe(2);
    expect(plan.expectedImageCount).toBe(6);
  });

  it('keeps every zipped group the same length, as the backend requires', () => {
    for (const seedBehaviour of ['per-iteration', 'per-image'] as const) {
      for (const seedStep of [-1, 0, 1] as const) {
        const plan = buildGeneratePromptBatchPlan(baseInput({ batchCount: 4, prompts, seedBehaviour, seedStep }));

        for (const group of plan.data) {
          const lengths = new Set(group.map((datum) => datum.items.length));

          expect(lengths.size).toBe(1);
        }
      }
    }
  });

  it('wraps seeds past the inclusive 32-bit ceiling', () => {
    const plan = buildGeneratePromptBatchPlan(
      baseInput({
        batchCount: 1,
        prompts: ['a', 'b', 'c'],
        seed: SEED_MAX - 1,
        seedBehaviour: 'per-image',
        seedStep: 1,
      })
    );

    expect(plan.data[0][0].items).toEqual([SEED_MAX - 1, SEED_MAX, 0]);
  });

  it('falls back to an empty prompt rather than emitting an empty batch', () => {
    const plan = buildGeneratePromptBatchPlan(baseInput({ prompts: [] }));

    expect(plan.data[0][1].items).toEqual(['']);
    expect(plan.expectedImageCount).toBe(1);
  });
});

describe('buildQueueWorkflowBatchPlan', () => {
  const seed = { fieldName: 'seed', nodeId: 'noise', seed: 10, seedStep: 1 as const };
  const cfg = { fieldName: 'cfg', items: [1, 2], nodeId: 'denoise' };
  const prompt = { fieldName: 'text', items: ['a', 'b', 'c'], nodeId: 'prompt' };
  const image = {
    fieldName: 'image',
    items: [{ image_name: 'x.png' }, { image_name: 'y.png' }, { image_name: 'z.png' }],
    nodeId: 'sink',
  };

  it('repeats one graph, or lets the backend multiply the groups, while every seed holds', () => {
    expect(buildQueueWorkflowBatchPlan({ batchCount: 3, batchData: undefined, seeds: [] })).toEqual({ runs: 3 });
    expect(
      buildQueueWorkflowBatchPlan({ batchCount: 2, batchData: [[cfg], [prompt, image]], seeds: undefined })
    ).toEqual({
      data: [
        [{ field_name: 'cfg', items: [1, 2], node_path: 'denoise' }],
        [
          { field_name: 'text', items: ['a', 'b', 'c'], node_path: 'prompt' },
          { field_name: 'image', items: image.items, node_path: 'sink' },
        ],
      ],
      runs: 2,
    });
  });

  it('keeps the seed-only plan: one zipped group of seeds over the runs', () => {
    expect(buildQueueWorkflowBatchPlan({ batchCount: 3, batchData: [], seeds: [seed] })).toEqual({
      data: [[{ field_name: 'seed', items: [10, 11, 12], node_path: 'noise' }]],
      runs: 1,
    });
  });

  it('gives every session its own seed by expanding the product run-major into one zipped group', () => {
    const plan = buildQueueWorkflowBatchPlan({ batchCount: 2, batchData: [[cfg], [prompt, image]], seeds: [seed] });

    expect(plan.runs).toBe(1);
    expect(plan.data).toHaveLength(1);

    const group = plan.data?.[0] ?? [];
    const byField = Object.fromEntries(group.map((datum) => [datum.field_name, datum.items]));

    // 2 runs × (2 cfg × 3 prompts) = 12 sessions, seeds walking straight through them.
    expect(byField.seed).toEqual([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]);
    // The first group varies slowest within a run, the run repeats the whole product.
    expect(byField.cfg).toEqual([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]);
    expect(byField.text).toEqual(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']);
    expect(byField.image).toEqual([...image.items, ...image.items, ...image.items, ...image.items]);
    expect(new Set(group.map((datum) => datum.items.length)).size).toBe(1);
  });

  it('recognises a persisted batch datum and rejects malformed ones', () => {
    expect(isQueueWorkflowBatchDatum(cfg)).toBe(true);
    expect(isQueueWorkflowBatchDatum(image)).toBe(true);
    expect(isQueueWorkflowBatchDatum({ ...cfg, items: [] })).toBe(false);
    expect(isQueueWorkflowBatchDatum({ ...cfg, items: [Number.NaN] })).toBe(false);
    expect(isQueueWorkflowBatchDatum({ ...cfg, items: [{ image_name: '' }] })).toBe(false);
    expect(isQueueWorkflowBatchDatum({ ...cfg, nodeId: '' })).toBe(false);
    expect(isQueueWorkflowBatchDatum({ ...cfg, items: Array.from({ length: 10_001 }, () => 1) })).toBe(false);
  });
});
