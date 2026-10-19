import { describe, expect, it } from 'vitest';

import { parsePreviewMetadata } from './previewMetadata';

describe('parsePreviewMetadata', () => {
  it('parses the common generate metadata shape', () => {
    const entries = parsePreviewMetadata({
      cfg_scale: 7.5,
      clip_skip: 1,
      height: 1024,
      model: { key: 'abc123', name: 'Dreamshaper XL' },
      negative_prompt: 'blurry',
      positive_prompt: 'a lighthouse at dusk',
      scheduler: 'euler',
      seed: 1234567,
      steps: 30,
      width: 768,
    });
    const byKey = Object.fromEntries(entries.map((entry) => [entry.key, entry.value]));

    expect(byKey).toEqual({
      cfgScale: '7.5',
      clipSkip: '1',
      model: 'Dreamshaper XL',
      negativePrompt: 'blurry',
      positivePrompt: 'a lighthouse at dusk',
      scheduler: 'euler',
      seed: '1234567',
      size: '768 x 1024',
      steps: '30',
    });
    expect(entries.find((entry) => entry.key === 'positivePrompt')?.isMultiline).toBe(true);
  });

  it('skips missing and malformed fields instead of throwing', () => {
    const entries = parsePreviewMetadata({
      cfg_scale: 'not-a-number',
      model: 'not-a-record',
      negative_prompt: null,
      positive_prompt: '',
      seed: Number.NaN,
      width: 512,
    });

    expect(entries).toEqual([]);
  });

  it('falls back to the model key when the name is missing', () => {
    const entries = parsePreviewMetadata({ model: { key: 'abc123' } });

    expect(entries).toEqual([{ key: 'model', label: 'Model', value: 'abc123' }]);
  });

  it('returns no entries for null metadata', () => {
    expect(parsePreviewMetadata(null)).toEqual([]);
  });
});
