import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { QueueItemReadModel } from '@features/queue/contracts';

import { buildQueueItemOrigin } from '@features/queue/contracts';
import { createDefaultVideoWidgetValues } from '@features/video';
import { describe, expect, it } from 'vitest';

import {
  buildQueueRecallValues,
  buildVideoQueueRecallPatch,
  getQueueRecallAvailability,
  getQueueRecallCapabilities,
  getQueueRecallSource,
  getVideoQueueRecallCapabilities,
  planQueueRecall,
  readQueueItemRecallInput,
} from './queueRecall';

const makeValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues =>
  ({
    aspectRatioId: '1:1',
    aspectRatioIsLocked: true,
    aspectRatioValue: 1,
    clipSkip: 0,
    expandPromptModelKey: null,
    imageToPromptModelKey: null,
    height: 1024,
    negativePrompt: '',
    negativePromptEnabled: false,
    positivePrompt: 'current prompt',
    seed: 1,
    seedMode: 'random',
    width: 1024,
    ...overrides,
  }) as GenerateWidgetValues;

describe('getQueueRecallAvailability', () => {
  const available = (input: Partial<Parameters<typeof getQueueRecallAvailability>[0]>) =>
    getQueueRecallAvailability({
      meta: {},
      snapshot: null,
      source: 'unknown',
      target: 'ready',
      videoSnapshot: null,
      ...input,
    });
  const SETTINGS_NOT_KEPT = 'widgets.queue.recallSettingsNotKept';

  it('offers an item without a snapshot in this browser only what the server recorded', () => {
    expect(available({ meta: { positivePrompt: 'p' } })).toEqual({
      capabilities: expect.objectContaining({ all: false, prompts: true, remix: false, seed: false }),
      reasons: {
        all: SETTINGS_NOT_KEPT,
        clipSkip: SETTINGS_NOT_KEPT,
        dimensions: SETTINGS_NOT_KEPT,
        remix: SETTINGS_NOT_KEPT,
        seed: SETTINGS_NOT_KEPT,
      },
    });
  });

  it.each([
    ['a Workflow or Upscale snapshot', getQueueRecallSource('workflow', false)],
    ['a workflow run another browser queued', getQueueRecallSource(null, true)],
  ])('offers nothing from %s, whatever its session values look like', (_, source) => {
    // Session values matched by type alone would read a workflow's text field as a prompt.
    const result = available({ meta: { positivePrompt: 'a workflow string field', seed: 3 }, source });

    expect(Object.values(result.capabilities).some(Boolean)).toBe(false);
    expect(new Set(Object.values(result.reasons))).toEqual(new Set(['widgets.queue.recallUnavailableForItem']));
    expect(Object.keys(result.reasons)).toHaveLength(6);
  });

  it('lets a video item recall into the Video panel whatever Generate holds', () => {
    const result = available({
      source: 'video',
      target: 'no-capabilities',
      videoSnapshot: { ...createDefaultVideoWidgetValues(), seedMode: 'fixed' },
    });

    expect(result.capabilities).toMatchObject({ all: true, prompts: true, seed: true });
    expect(result.reasons).toEqual({
      clipSkip: 'widgets.queue.recallNotForVideo',
      dimensions: 'widgets.queue.recallNotForVideo',
    });
  });
});

describe('readQueueItemRecallInput', () => {
  const field = (nodePath: string, value: number | string) => ({ fieldName: 'value', nodePath, value });
  const item = (overrides: Partial<QueueItemReadModel>): QueueItemReadModel => ({
    batchId: 'batch',
    createdAt: '2026-01-01T00:00:00Z',
    id: 1,
    resultImageNames: [],
    sessionId: 'session',
    status: 'completed',
    updatedAt: '2026-01-01T00:00:00Z',
    ...overrides,
  });
  const webv2Origin = buildQueueItemOrigin('run', 'project');

  it('reads a webv2 Generate-style run by its named prompt and seed only', () => {
    expect(
      readQueueItemRecallInput(
        item({
          fieldValues: [field('caption', 'not a prompt'), field('positive_prompt', 'a fox'), field('seed', 7)],
          origin: webv2Origin,
        })
      )
    ).toEqual({ isNonGenerateRun: false, meta: { positivePrompt: 'a fox', seed: 7 } });
  });

  it("still reads another client's unfamiliar node names by type and order", () => {
    expect(readQueueItemRecallInput(item({ fieldValues: [field('prompt_node', 'a fox'), field('noise', 3)] }))).toEqual(
      { isNonGenerateRun: false, meta: { positivePrompt: 'a fox', seed: 3 } }
    );
  });
});

describe('getQueueRecallSource', () => {
  it.each([
    ['generate', false, 'generate'],
    ['canvas', false, 'generate'],
    ['video', true, 'video'],
    ['upscale', false, 'other'],
    [null, true, 'other'],
    [null, false, 'unknown'],
  ] as const)('reads a %s snapshot (workflow run named: %s) as %s', (snapshotSourceId, isWorkflowRun, expected) => {
    expect(getQueueRecallSource(snapshotSourceId, isWorkflowRun)).toBe(expected);
  });
});

describe('getQueueRecallCapabilities', () => {
  it('grants everything for local snapshots', () => {
    expect(getQueueRecallCapabilities(makeValues({ seedMode: 'fixed' }), {})).toEqual({
      all: true,
      clipSkip: true,
      dimensions: true,
      prompts: true,
      remix: true,
      seed: true,
      workflow: false,
    });
  });

  it('grants prompts and seed for foreign items with session meta', () => {
    expect(getQueueRecallCapabilities(null, { positivePrompt: 'p', seed: 42 })).toEqual({
      all: false,
      clipSkip: false,
      dimensions: false,
      prompts: true,
      remix: false,
      seed: true,
      workflow: false,
    });
  });
});

describe('buildQueueRecallValues', () => {
  const current = makeValues();

  it('recalls all as the exact snapshot and remix with a randomized seed', () => {
    const snapshot = makeValues({ positivePrompt: 'snap', seedMode: 'fixed' });

    expect(buildQueueRecallValues('all', { current, meta: {}, snapshot })).toEqual(snapshot);
    expect(buildQueueRecallValues('remix', { current, meta: {}, snapshot })).toEqual({
      ...snapshot,
      seedMode: 'random',
    });
  });

  it('recalls prompts from session meta for foreign items', () => {
    const result = buildQueueRecallValues('prompts', {
      current,
      meta: { negativePrompt: 'meta neg', positivePrompt: 'meta' },
      snapshot: null,
    });

    expect(result).toEqual({
      ...current,
      negativePrompt: 'meta neg',
      negativePromptEnabled: true,
      positivePrompt: 'meta',
      promptTemplate: null,
    });
  });

  it('prefers the executed session seed and pins randomization off', () => {
    const snapshot = makeValues({ seed: 7, seedMode: 'fixed' });

    expect(buildQueueRecallValues('seed', { current, meta: { seed: 42 }, snapshot })).toEqual({
      ...current,
      seed: 42,
      seedMode: 'fixed',
    });
    expect(buildQueueRecallValues('seed', { current, meta: {}, snapshot })).toEqual({
      ...current,
      seed: 7,
      seedMode: 'fixed',
    });
    expect(buildQueueRecallValues('seed', { current, meta: {}, snapshot: null })).toBeNull();
  });
});

describe('buildVideoQueueRecallPatch', () => {
  it('recalls exact video settings or a randomized remix without crossing into Generate', () => {
    const snapshot = {
      ...createDefaultVideoWidgetValues(),
      positivePrompt: 'snapshot prompt',
      seed: 123,
      seedMode: 'fixed' as const,
    };
    expect(getVideoQueueRecallCapabilities(snapshot, {})).toEqual({
      all: true,
      remix: true,
      prompts: true,
      seed: true,
      dimensions: false,
      clipSkip: false,
      workflow: false,
    });
    expect(
      planQueueRecall('all', { current: null, isVideoItem: true, meta: {}, snapshot: null, videoSnapshot: snapshot })
    ).toEqual({ target: 'video', patch: snapshot });
    expect(buildVideoQueueRecallPatch('remix', {}, snapshot)).toEqual({ ...snapshot, seedMode: 'random' });
    expect(snapshot.seedMode).toBe('fixed');
    expect(buildVideoQueueRecallPatch('seed', { seed: 456 }, snapshot)).toEqual({
      seed: 456,
      seedMode: 'fixed',
    });
  });

  it('patches only the prompt keys, so the rest of the Video panel is untouched', () => {
    expect(
      buildVideoQueueRecallPatch('prompts', { negativePrompt: 'blurry', positivePrompt: 'a fox running' })
    ).toEqual({ negativePrompt: 'blurry', negativePromptEnabled: true, positivePrompt: 'a fox running' });
  });
});

describe('planQueueRecall', () => {
  const current = makeValues();
  const meta = { negativePrompt: 'blurry', positivePrompt: 'a fox running', seed: 4321 };

  it('targets Video for an item this client submitted from Video', () => {
    // The bug this replaced: a video item's prompt was written into Generate.
    expect(planQueueRecall('prompts', { current, isVideoItem: true, meta, snapshot: null })).toEqual({
      patch: { negativePrompt: 'blurry', negativePromptEnabled: true, positivePrompt: 'a fox running' },
      target: 'video',
    });
    expect(planQueueRecall('seed', { current, isVideoItem: true, meta, snapshot: null })).toEqual({
      patch: { seed: 4321, seedMode: 'fixed' },
      target: 'video',
    });
  });
});
