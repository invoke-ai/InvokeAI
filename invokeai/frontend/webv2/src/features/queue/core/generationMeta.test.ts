import { describe, expect, it } from 'vitest';

import type { QueueItemReadModel, QueueNodeFieldValue } from './types';

import { extractGenerationMeta } from './generationMeta';

const item = (fieldValues: QueueNodeFieldValue[]): QueueItemReadModel => ({ fieldValues }) as QueueItemReadModel;

const field = (nodePath: string, value: string | number): QueueNodeFieldValue => ({
  fieldName: 'value',
  nodePath,
  value,
});

describe('extractGenerationMeta', () => {
  it('reads the prompts and seed by node identity', () => {
    expect(
      extractGenerationMeta(
        item([field('seed', 42), field('positive_prompt', 'a fox running'), field('negative_prompt', 'blurry')])
      )
    ).toEqual({ negativePrompt: 'blurry', positivePrompt: 'a fox running', seed: 42 });
  });

  it('keeps an empty positive prompt empty instead of promoting the negative', () => {
    // Preserve empty named positive prompts; positional scanning must not misread negative text as the video
    // subject.
    const meta = extractGenerationMeta(
      item([field('seed', 7), field('positive_prompt', ''), field('negative_prompt', 'blurry, low quality')])
    );

    expect(meta).toEqual({ negativePrompt: 'blurry, low quality', positivePrompt: '', seed: 7 });
  });

  it('falls back to value type and order for graphs that name their own nodes', () => {
    expect(
      extractGenerationMeta(item([field('noise', 99), field('prompt_a', 'subject'), field('prompt_b', 'avoid')]))
    ).toEqual({ negativePrompt: 'avoid', positivePrompt: 'subject', seed: 99 });
  });

  it('recovers a seed by type when only the prompts are named', () => {
    expect(extractGenerationMeta(item([field('noise', 5), field('positive_prompt', 'a fox')]))).toEqual({
      positivePrompt: 'a fox',
      seed: 5,
    });
  });

  it('returns an empty meta for an item with no field values', () => {
    expect(extractGenerationMeta({} as QueueItemReadModel)).toEqual({});
  });
});
