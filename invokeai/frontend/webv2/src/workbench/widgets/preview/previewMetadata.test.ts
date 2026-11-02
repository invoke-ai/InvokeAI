import type { TFunction } from 'i18next';

import i18n from 'i18next';
import { beforeAll, describe, expect, it } from 'vitest';

import { parsePreviewMetadata } from './previewMetadata';

const englishCatalog = Object.values(
  import.meta.glob('../../../../public/locales/en.json', { eager: true, import: 'default' })
)[0] as Record<string, unknown>;
const instance = i18n.createInstance();
const t = ((key: string) => instance.t(key)) as TFunction;

beforeAll(async () => {
  await instance.init({
    initAsync: false,
    interpolation: { escapeValue: false },
    lng: 'en',
    resources: { en: { translation: englishCatalog } },
  });
});

const image = { kind: 'image', t } as const;
const video = { kind: 'video', t } as const;
const values = (entries: ReturnType<typeof parsePreviewMetadata>) =>
  Object.fromEntries(entries.map((entry) => [entry.key, entry.value]));

describe('parsePreviewMetadata', () => {
  it('parses the common generate metadata shape under localized labels', () => {
    const entries = parsePreviewMetadata(
      {
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
      },
      image
    );

    expect(values(entries)).toEqual({
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
    expect(entries.map((entry) => entry.label)).toEqual([
      'Prompt',
      'Negative Prompt',
      'Model',
      'Seed',
      'Steps',
      'CFG Scale',
      'Scheduler',
      'CLIP Skip',
      'Size',
    ]);
    expect(entries.find((entry) => entry.key === 'positivePrompt')?.isMultiline).toBe(true);
  });

  it('skips missing and malformed fields instead of throwing', () => {
    const entries = parsePreviewMetadata(
      {
        cfg_scale: 'not-a-number',
        model: 'not-a-record',
        negative_prompt: null,
        positive_prompt: '',
        seed: Number.NaN,
        width: 512,
      },
      image
    );

    expect(entries).toEqual([]);
  });

  it('names the model from a pre-identifier record, or by its key when no name was recorded', () => {
    expect(values(parsePreviewMetadata({ model: { base_model: 'sd-1', model_name: 'Old Model' } }, image))).toEqual({
      model: 'Old Model',
    });
    expect(parsePreviewMetadata({ model: { key: 'abc123' } }, image)).toEqual([
      { key: 'model', label: 'Model', value: 'abc123' },
    ]);
  });

  it('shows the guidance a record stores without a CFG scale, whatever the model', () => {
    expect(parsePreviewMetadata({ guidance: 3.5, model: { name: 'Unknown' } }, image)).toContainEqual({
      key: 'guidance',
      label: 'Guidance',
      value: '3.5',
    });
    expect(values(parsePreviewMetadata({ cfg_scale: 1, guidance: 3.5 }, image))).toEqual({ cfgScale: '1' });
  });

  it('parses video rows: frames and FPS instead of the image-only fields', () => {
    const entries = parsePreviewMetadata(
      {
        cfg_scale: 5,
        clip_skip: 2,
        fps: 16,
        guidance: 3,
        height: 720,
        model: { base: 'wan', key: 'wan-key', name: 'Wan 2.2 T2V', type: 'main' },
        negative_prompt: 'blurry',
        num_frames: 81,
        positive_prompt: 'a red fox',
        scheduler: 'euler',
        seed: 7,
        steps: 30,
        width: 1280,
      },
      video
    );

    expect(values(entries)).toEqual({
      cfgScale: '5',
      fps: '16',
      frames: '81',
      model: 'Wan 2.2 T2V',
      negativePrompt: 'blurry',
      positivePrompt: 'a red fox',
      seed: '7',
      size: '1280 x 720',
      steps: '30',
    });
    expect(entries.find((entry) => entry.key === 'frames')?.label).toBe('Frames');
    expect(entries.find((entry) => entry.key === 'fps')?.label).toBe('FPS');
  });

  it('returns no entries for null metadata', () => {
    expect(parsePreviewMetadata(null, image)).toEqual([]);
    expect(parsePreviewMetadata(null, video)).toEqual([]);
  });
});
