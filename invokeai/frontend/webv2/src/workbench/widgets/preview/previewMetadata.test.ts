import type { TFunction } from 'i18next';

import i18n from 'i18next';
import { beforeAll, describe, expect, it } from 'vitest';

import { parsePreviewMetadata } from './previewMetadata';

const englishCatalog = Object.values(
  import.meta.glob('../../../../public/locales/en.json', { eager: true, import: 'default' })
)[0] as Record<string, unknown>;
const instance = i18n.createInstance();
const t = instance.t.bind(instance) as TFunction;

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

  it('keeps basic rows while omitting extra rows whose labels have not loaded', () => {
    const unavailableLabels = ((key: string) =>
      key.startsWith('widgets.preview.metadataLabels.') ? key : t(key)) as TFunction;
    expect(
      parsePreviewMetadata(
        {
          seed: 17,
          strength: 0.4,
          vae: { name: 'VAE' },
          ideogram4_color_palette: ['green'],
          control_layers: [{ model: { name: 'Depth model' } }],
          canvas_v3_metadata: { document: { stacks: { control: [{ id: 'control', name: 'Depth map' }] } } },
        },
        { kind: 'image', t: unavailableLabels }
      )
    ).toEqual([{ key: 'seed', label: 'Seed', value: '17' }]);
  });
});

it('shows useful controls, authored layers, adapters and family settings with raw records left intact', () => {
  const metadata = {
    strength: 0.4,
    refiner_scheduler: 'euler',
    wan_guidance_scale_low_noise: 2.5,
    ideogram4_color_palette: ['green', 'gold'],
    ref_images: [
      { config: { type: 'ip_adapter', model: { name: 'Style' }, image: { image_name: 'style.png' }, weight: 0.6 } },
    ],
    control_layers: [
      { model: { name: 'Edges' }, image: { image_name: 'edges.png' }, type: 'controlnet', weight: 0.75 },
    ],
    canvas_v3_metadata: {
      document: {
        stacks: {
          control: [{ id: 'c', type: 'control', name: 'My edges', adapter: { model: 'control-key', weight: 0.75 } }],
        },
      },
      controlModels: { c: { name: 'Edges' } },
    },
  };
  const entries = parsePreviewMetadata(metadata, image);
  expect(values(entries)).toMatchObject({
    strength: '0.4',
    refiner_scheduler: 'euler',
    wan_guidance_scale_low_noise: '2.5',
    ideogram4_color_palette: 'green, gold',
    'control_layers-0': 'Edges · controlnet · edges.png · Weight 0.75',
    'ref_images-0': 'Style · ip_adapter · style.png · Weight 0.6',
    canvasLayerCount: '1 layer',
  });
  expect(entries.some((entry) => entry.value.includes('My edges') && entry.value.includes('Edges'))).toBe(true);
  expect(metadata.canvas_v3_metadata.document.stacks.control[0]?.id).toBe('c');
});

it('describes v4 IP-Adapters including model, image, vision, method, weight and schedule', () => {
  const entry = {
    ip_adapter_model: { model_name: 'Portrait', base_model: 'sdxl' },
    image: { image_name: 'portrait.png' },
    clip_vision_model: 'ViT-G',
    method: 'style_precise',
    weight: 0.65,
    begin_step_percent: 0.2,
    end_step_percent: 0.9,
  };
  const parsed = parsePreviewMetadata({ ipAdapters: [entry] }, image);
  expect(parsed).toHaveLength(1);
  expect(parsed[0]?.value).toBe('Portrait · portrait.png · ViT-G · style_precise · Weight 0.65 · Steps 0.2–0.9');
  expect(parsePreviewMetadata({ ref_images: [], ipAdapters: [entry] }, image)).toEqual([]);
});

it.each(['array', 'entities'] as const)(
  'summarizes historical global references in a %s without counting them as Canvas layers',
  (shape) => {
    const reference = {
      id: 'global-style',
      type: 'reference_image',
      name: 'Global style',
      isEnabled: true,
      ipAdapter: {
        type: 'ip_adapter',
        model: { key: 'style', name: 'Historical style', base: 'sdxl', type: 'ip_adapter' },
        image: { image_name: 'style.png', width: 320, height: 200 },
        weight: 0.6,
        method: 'composition',
        clipVisionModel: 'ViT-L',
        beginEndStepPct: [0.1, 0.8],
      },
    };
    const canvas = {
      rasterLayers: [{ id: 'raster', type: 'raster_layer', name: 'Authored raster' }],
      controlLayers: [],
      inpaintMasks: [],
      regionalGuidance: [],
      referenceImages: shape === 'array' ? [reference] : { entities: [reference] },
    };
    const metadata = {
      canvas_v2_metadata: canvas,
      ipAdapters: [{ ip_adapter_model: { name: 'Duplicate alias' }, image: { image_name: 'style.png' }, weight: 0.6 }],
    };
    const parsed = parsePreviewMetadata(metadata, image);
    expect(values(parsed)).toMatchObject({
      'ref_images-0': 'Historical style · ip_adapter · style.png · ViT-L · composition · Weight 0.6 · Steps 0.1–0.8',
      canvasLayerCount: '1 layer',
    });
    expect(parsed.find((entry) => entry.key === 'ref_images-0')?.label).toBe('Reference images');
    expect(parsed.filter((entry) => entry.value.includes('Historical style'))).toHaveLength(1);
    expect(
      parsed.some((entry) => entry.value.includes('Duplicate alias') || entry.value.includes('Global style'))
    ).toBe(false);
    for (const ref_images of [
      [],
      [
        {
          id: 'modern',
          config: {
            type: 'ip_adapter',
            model: { name: 'Modern style' },
            image: { image_name: 'modern.png' },
            weight: 0.4,
          },
        },
      ],
    ]) {
      const modern = parsePreviewMetadata({ ...metadata, ref_images }, image);
      expect(
        modern.some((entry) => entry.value.includes('Historical style') || entry.value.includes('Duplicate alias'))
      ).toBe(false);
      expect(modern.filter((entry) => entry.key.startsWith('ref_images-'))).toHaveLength(ref_images.length);
      expect(values(modern).canvasLayerCount).toBe('1 layer');
    }
  }
);
