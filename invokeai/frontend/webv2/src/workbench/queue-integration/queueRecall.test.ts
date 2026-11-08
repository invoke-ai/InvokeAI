import type { GenerateWidgetValues } from '@features/generation/contracts';

import { createDefaultVideoWidgetValues } from '@features/video';
import { describe, expect, it } from 'vitest';

import {
  buildQueueRecallValues,
  buildVideoQueueRecallPatch,
  getQueueRecallCapabilities,
  getVideoQueueRecallCapabilities,
  planQueueRecall,
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
