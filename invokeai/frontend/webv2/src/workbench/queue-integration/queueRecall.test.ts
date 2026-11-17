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

  it('names what the Generate panel lacks beside an available full recall', () => {
    const snapshot = makeValues({ seedMode: 'fixed' });

    expect(available({ snapshot, source: 'generate', target: 'no-model' })).toEqual({
      capabilities: expect.objectContaining({ all: true, dimensions: false, prompts: false, remix: true }),
      reasons: {
        clipSkip: 'widgets.generate.selectSupportedModelFirst',
        dimensions: 'widgets.generate.selectSupportedModelFirst',
        prompts: 'widgets.generate.selectSupportedModelFirst',
        seed: 'widgets.generate.selectSupportedModelFirst',
      },
    });
    expect(available({ snapshot, source: 'generate', target: 'no-capabilities' }).reasons.prompts).toBe(
      'widgets.generate.capabilitiesUnavailableForSetup'
    );
  });

  it('says nothing is recallable while partial kinds wait and nothing else can be recalled', () => {
    const meta = { positivePrompt: 'p' };

    expect(available({ meta, target: 'no-model' }).reasons.prompts).toBe('widgets.queue.recallUnavailable');
    expect(available({ meta, target: 'no-capabilities' }).reasons.prompts).toBe(
      'widgets.queue.recallUnavailableCapabilities'
    );
  });

  it('gives no reason while the model catalog is still loading', () => {
    const result = available({ meta: { positivePrompt: 'p' }, target: 'loading' });

    expect(result.capabilities.prompts).toBe(false);
    expect(result.reasons.prompts).toBeUndefined();
  });

  it('says a random seed is unreported rather than unrecorded', () => {
    expect(available({ snapshot: makeValues({ seedMode: 'random' }), source: 'generate' }).reasons.seed).toBe(
      'widgets.queue.recallRandomSeed'
    );
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

  it.each([
    ['a webv2 run without the named values', { fieldValues: [field('caption', 'a caption'), field('n', 3)] }],
    // Several workflow_return nodes submit no workflow, so the server names none.
    ['a webv2 workflow that submitted no workflow', { fieldValues: [] }],
  ])('treats %s as something other than Generate', (_, overrides) => {
    expect(readQueueItemRecallInput(item({ ...overrides, origin: webv2Origin }))).toEqual({
      isNonGenerateRun: true,
      meta: {},
    });
  });

  it('treats a workflow with an empty name as a workflow run', () => {
    expect(readQueueItemRecallInput(item({ workflowName: '' })).isNonGenerateRun).toBe(true);
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

  it('withholds seed for randomized local submissions without session meta', () => {
    expect(getQueueRecallCapabilities(makeValues({ seedMode: 'random' }), {}).seed).toBe(false);
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

  it('keeps the current prompt tool model picks when restoring a snapshot', () => {
    const picked = makeValues({ expandPromptModelKey: 'llm-b', imageToPromptModelKey: 'vision-b' });
    const snapshot = makeValues({
      expandPromptModelKey: null,
      imageToPromptModelKey: 'vision-a',
      positivePrompt: 'snap',
    });

    for (const kind of ['all', 'remix'] as const) {
      expect(buildQueueRecallValues(kind, { current: picked, meta: {}, snapshot })).toMatchObject({
        expandPromptModelKey: 'llm-b',
        imageToPromptModelKey: 'vision-b',
        positivePrompt: 'snap',
      });
    }
  });

  it('merges prompts into the current values, preferring the snapshot', () => {
    const snapshot = makeValues({ negativePrompt: 'snap neg', negativePromptEnabled: true, positivePrompt: 'snap' });
    const result = buildQueueRecallValues('prompts', { current, meta: { positivePrompt: 'meta' }, snapshot });

    expect(result).toEqual({
      ...current,
      negativePrompt: 'snap neg',
      negativePromptEnabled: true,
      positivePrompt: 'snap',
      promptTemplate: null,
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

  // Session metadata carries the prompt the model was given, template already
  // applied, so recalling it has to stop the active template wrapping it again.
  it('clears the active prompt template when recalling prompts from session meta', () => {
    const withTemplate = makeValues({
      promptTemplate: { id: 't1', name: 'Cinematic', negativePrompt: '', positivePrompt: '{prompt}, cinematic' },
    });

    expect(
      buildQueueRecallValues('prompts', { current: withTemplate, meta: { positivePrompt: 'meta' }, snapshot: null })
        ?.promptTemplate
    ).toBeNull();
  });

  // Snapshots store authored prompts, so recall must restore their template too.
  it('recalls the snapshot`s own template alongside its authored prompt', () => {
    const promptTemplate = { id: 't1', name: 'Cinematic', negativePrompt: '', positivePrompt: '{prompt}, cinematic' };
    const snapshot = makeValues({ positivePrompt: 'a cat', promptTemplate });

    const result = buildQueueRecallValues('prompts', {
      current: makeValues(),
      meta: { positivePrompt: 'a cat, cinematic' },
      snapshot,
    });

    expect(result?.positivePrompt).toBe('a cat');
    expect(result?.promptTemplate).toEqual(promptTemplate);
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

  it('recalls dimensions with the snapshot aspect state', () => {
    const snapshot = makeValues({ aspectRatioId: '3:4', aspectRatioValue: 0.75, height: 1152, width: 896 });

    expect(buildQueueRecallValues('dimensions', { current, meta: {}, snapshot })).toEqual({
      ...current,
      aspectRatioId: '3:4',
      aspectRatioIsLocked: true,
      aspectRatioValue: 0.75,
      height: 1152,
      width: 896,
    });
  });

  it('returns null for partial kinds without current form values', () => {
    expect(
      buildQueueRecallValues('prompts', { current: null, meta: { positivePrompt: 'p' }, snapshot: null })
    ).toBeNull();
    expect(buildQueueRecallValues('seed', { current: null, meta: { seed: 1 }, snapshot: null })).toBeNull();
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

  it('leaves the negative toggle alone when no usable negative was recorded', () => {
    expect(buildVideoQueueRecallPatch('prompts', { positivePrompt: 'a fox running' })).toEqual({
      positivePrompt: 'a fox running',
    });
    expect(buildVideoQueueRecallPatch('prompts', { negativePrompt: '', positivePrompt: 'a fox running' })).toEqual({
      positivePrompt: 'a fox running',
    });
  });

  it('patches the executed seed and stops it being re-randomised', () => {
    expect(buildVideoQueueRecallPatch('seed', { seed: 4321 })).toEqual({ seed: 4321, seedMode: 'fixed' });
  });

  it('returns null when the meta carries nothing for the verb', () => {
    expect(buildVideoQueueRecallPatch('prompts', {})).toBeNull();
    expect(buildVideoQueueRecallPatch('seed', {})).toBeNull();
  });

  it('returns null for the verbs a video queue item never offers', () => {
    const meta = { positivePrompt: 'a fox running', seed: 1 };

    // These are disabled by `getQueueRecallCapabilities` for a video item (its
    // Generate-shaped snapshot is always null), so they must not half-apply.
    for (const kind of ['all', 'remix', 'dimensions', 'clipSkip'] as const) {
      expect(buildVideoQueueRecallPatch(kind, meta)).toBeNull();
    }
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

  it('targets Generate for a non-video item, and for one whose source is unknown', () => {
    const plan = planQueueRecall('prompts', { current, isVideoItem: false, meta, snapshot: null });

    expect(plan?.target).toBe('generate');
    expect(plan).toMatchObject({ values: { positivePrompt: 'a fox running' } });
  });

  it('returns null rather than falling through to the other panel', () => {
    // A verb the video branch cannot serve must not quietly recall into Generate.
    expect(planQueueRecall('all', { current, isVideoItem: true, meta, snapshot: makeValues() })).toBeNull();
    expect(planQueueRecall('prompts', { current, isVideoItem: true, meta: {}, snapshot: null })).toBeNull();
  });
});
