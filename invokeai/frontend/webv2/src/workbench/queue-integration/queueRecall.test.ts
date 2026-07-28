import type { GenerateWidgetValues } from '@features/generation/contracts';

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

  // A snapshot is the other story: it stores the prompt as *authored*, so the
  // template that shaped it has to come back too. Recalling `a cat` and dropping
  // `Cinematic` silently generated something other than the item recalled from.
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
