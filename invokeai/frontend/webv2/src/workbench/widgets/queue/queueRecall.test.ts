import type { GenerateWidgetValues } from '@workbench/generation/types';

import { describe, expect, it } from 'vitest';

import { buildQueueRecallValues, getQueueRecallCapabilities } from './queueRecall';

const makeValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues =>
  ({
    aspectRatioId: '1:1',
    aspectRatioIsLocked: true,
    aspectRatioValue: 1,
    clipSkip: 0,
    height: 1024,
    negativePrompt: '',
    negativePromptEnabled: false,
    positivePrompt: 'current prompt',
    seed: 1,
    shouldRandomizeSeed: true,
    width: 1024,
    ...overrides,
  }) as GenerateWidgetValues;

describe('getQueueRecallCapabilities', () => {
  it('grants everything for local snapshots', () => {
    expect(getQueueRecallCapabilities(makeValues({ shouldRandomizeSeed: false }), {})).toEqual({
      all: true,
      clipSkip: true,
      dimensions: true,
      prompts: true,
      remix: true,
      seed: true,
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
    });
  });

  it('withholds seed for randomized local submissions without session meta', () => {
    expect(getQueueRecallCapabilities(makeValues({ shouldRandomizeSeed: true }), {}).seed).toBe(false);
  });
});

describe('buildQueueRecallValues', () => {
  const current = makeValues();

  it('recalls all as the exact snapshot and remix with a randomized seed', () => {
    const snapshot = makeValues({ positivePrompt: 'snap', shouldRandomizeSeed: false });

    expect(buildQueueRecallValues('all', { current, meta: {}, snapshot })).toBe(snapshot);
    expect(buildQueueRecallValues('remix', { current, meta: {}, snapshot })).toEqual({
      ...snapshot,
      shouldRandomizeSeed: true,
    });
  });

  it('merges prompts into the current values, preferring the snapshot', () => {
    const snapshot = makeValues({ negativePrompt: 'snap neg', negativePromptEnabled: true, positivePrompt: 'snap' });
    const result = buildQueueRecallValues('prompts', { current, meta: { positivePrompt: 'meta' }, snapshot });

    expect(result).toEqual({
      ...current,
      negativePrompt: 'snap neg',
      negativePromptEnabled: true,
      positivePrompt: 'snap',
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
    });
  });

  it('prefers the executed session seed and pins randomization off', () => {
    const snapshot = makeValues({ seed: 7, shouldRandomizeSeed: false });

    expect(buildQueueRecallValues('seed', { current, meta: { seed: 42 }, snapshot })).toEqual({
      ...current,
      seed: 42,
      shouldRandomizeSeed: false,
    });
    expect(buildQueueRecallValues('seed', { current, meta: {}, snapshot })).toEqual({
      ...current,
      seed: 7,
      shouldRandomizeSeed: false,
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
