import type { RootState } from 'app/store/store';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation } from 'services/api/types';
import { describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  getShouldProcessPrompt: vi.fn((prompt: string) => prompt.includes('__')),
  selectPresetModifiedPrompts: vi.fn((state: RootState) => ({ positive: state.params.positivePrompt })),
}));

vi.mock('features/dynamicPrompts/util/getShouldProcessPrompt', () => ({
  getShouldProcessPrompt: mocks.getShouldProcessPrompt,
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  selectPresetModifiedPrompts: mocks.selectPresetModifiedPrompts,
}));

import { prepareLinearUIBatch } from './buildLinearBatchConfig';

const g = {
  getGraph: () => ({ id: 'graph', nodes: {}, edges: [] }),
} as unknown as Graph;

const positivePromptNode = { id: 'positive_prompt' } as Invocation<'string'>;
const seedNode = { id: 'seed' } as Invocation<'integer'>;

const buildState = (overrides: {
  iterations?: number;
  positivePrompt?: string;
  prompts?: string[];
  randomRefreshMode?: 'manual' | 'per_enqueue' | 'per_image';
  randomSamples?: number;
  seedBehaviour?: 'PER_ITERATION' | 'PER_PROMPT';
}): RootState =>
  ({
    params: {
      iterations: overrides.iterations ?? 3,
      shouldRandomizeSeed: false,
      seed: 100,
      positivePrompt: overrides.positivePrompt ?? '__lighting/studio__',
    },
    dynamicPrompts: {
      mode: 'random',
      randomRefreshMode: overrides.randomRefreshMode ?? 'per_image',
      randomSamples: overrides.randomSamples ?? 1,
      prompts: overrides.prompts ?? ['softbox', 'window light', 'rim light'],
      seedBehaviour: overrides.seedBehaviour ?? 'PER_ITERATION',
    },
  }) as RootState;

describe('prepareLinearUIBatch', () => {
  it('zips per-image random prompts with one seed per output', () => {
    const batch = prepareLinearUIBatch({
      state: buildState({ iterations: 3, prompts: ['softbox', 'window light', 'rim light'] }),
      g,
      prepend: false,
      base: 'sdxl',
      positivePromptNode,
      seedNode,
      origin: 'generate',
      destination: 'generate',
    }).batch;

    expect(batch.runs).toBe(1);
    expect(batch.data).toEqual([
      [
        { node_path: 'seed', field_name: 'value', items: [100, 101, 102] },
        { node_path: 'positive_prompt', field_name: 'value', items: ['softbox', 'window light', 'rim light'] },
      ],
    ]);
  });

  it('queues randomSamples times iterations outputs for per-image random prompts', () => {
    const prompts = ['a', 'b', 'c', 'd', 'e', 'f'];
    const batch = prepareLinearUIBatch({
      state: buildState({ iterations: 3, randomSamples: 2, prompts }),
      g,
      prepend: false,
      base: 'sdxl',
      positivePromptNode,
      seedNode,
      origin: 'generate',
      destination: 'generate',
    }).batch;

    expect(batch.data).toEqual([
      [
        { node_path: 'seed', field_name: 'value', items: [100, 101, 102, 103, 104, 105] },
        { node_path: 'positive_prompt', field_name: 'value', items: prompts },
      ],
    ]);
  });

  it('keeps static prompts on normal iteration batching even when per-image is selected', () => {
    const batch = prepareLinearUIBatch({
      state: buildState({ positivePrompt: 'portrait', prompts: ['portrait'], iterations: 3 }),
      g,
      prepend: false,
      base: 'sdxl',
      positivePromptNode,
      seedNode,
      origin: 'generate',
      destination: 'generate',
    }).batch;

    expect(batch.data).toEqual([
      [{ node_path: 'seed', field_name: 'value', items: [100, 101, 102] }],
      [{ node_path: 'positive_prompt', field_name: 'value', items: ['portrait'] }],
    ]);
  });

  it('keeps per-invoke random prompts on normal prompt by iteration batching', () => {
    const batch = prepareLinearUIBatch({
      state: buildState({ randomRefreshMode: 'per_enqueue', prompts: ['softbox'], iterations: 3 }),
      g,
      prepend: false,
      base: 'sdxl',
      positivePromptNode,
      seedNode,
      origin: 'generate',
      destination: 'generate',
    }).batch;

    expect(batch.data).toEqual([
      [{ node_path: 'seed', field_name: 'value', items: [100, 101, 102] }],
      [{ node_path: 'positive_prompt', field_name: 'value', items: ['softbox'] }],
    ]);
  });

  it('zips per-invoke cycle-only prompts with one seed per output', () => {
    const batch = prepareLinearUIBatch({
      state: buildState({
        randomRefreshMode: 'per_enqueue',
        positivePrompt: '__@lighting/studio__',
        prompts: ['softbox', 'window light', 'rim light'],
        iterations: 3,
      }),
      g,
      prepend: false,
      base: 'sdxl',
      positivePromptNode,
      seedNode,
      origin: 'generate',
      destination: 'generate',
    }).batch;

    expect(batch.data).toEqual([
      [
        { node_path: 'seed', field_name: 'value', items: [100, 101, 102] },
        { node_path: 'positive_prompt', field_name: 'value', items: ['softbox', 'window light', 'rim light'] },
      ],
    ]);
  });

  it('zips mixed cycle and random prompts with one seed per output', () => {
    const batch = prepareLinearUIBatch({
      state: buildState({
        randomRefreshMode: 'per_enqueue',
        positivePrompt: '__@lighting/studio__, __camera/lens__',
        prompts: ['softbox, 50mm', 'window light, 50mm', 'rim light, 50mm'],
        iterations: 3,
      }),
      g,
      prepend: false,
      base: 'sdxl',
      positivePromptNode,
      seedNode,
      origin: 'generate',
      destination: 'generate',
    }).batch;

    expect(batch.data).toEqual([
      [
        { node_path: 'seed', field_name: 'value', items: [100, 101, 102] },
        {
          node_path: 'positive_prompt',
          field_name: 'value',
          items: ['softbox, 50mm', 'window light, 50mm', 'rim light, 50mm'],
        },
      ],
    ]);
  });
});
