import type { GenerationModelCatalogItem, MainModelConfig } from '@features/generation/contracts';

import {
  compileVideoGraph,
  createDefaultVideoWidgetValues,
  getAcceleratorToggleResult,
  getDefaultVideoSettings,
  getVideoDimensions,
  syncVideoWidgetValuesWithModels,
  type VideoWidgetValues,
} from '@features/video';
import { describe, expect, it } from 'vitest';

import { recordOf } from './recallContract.testing';
import {
  buildVideoRecallSettings,
  deriveAcceleratorRecallState,
  getVideoRecallCapabilities,
  getVideoSizeRecalls,
  parseVideoRecallRecord,
  resolveVideoSizeRecall,
  settleMediaSizedRecall,
} from './videoRecall';

const wanModel = (variant: string, format = 'diffusers', key = `wan-${variant}-${format}`): MainModelConfig => ({
  base: 'wan',
  format,
  key,
  name: `Wan 2.2 ${variant}`,
  type: 'main',
  variant,
});

const h3Model = (key = 'h3-main'): MainModelConfig => ({
  base: 'minimax-h3',
  format: 'diffusers',
  key,
  name: 'MiniMax H3',
  type: 'main',
  variant: 'fl2va',
});

const lora = (name: string, base = 'wan', variant: string | null = 'a14b'): GenerationModelCatalogItem => ({
  base,
  key: `lora-${name}`,
  name,
  type: 'lora',
  variant,
});

const LIGHTNING_HIGH = lora('Wan 2.2 T2V Lightning High Noise (4-step)');
const LIGHTNING_LOW = lora('Wan 2.2 T2V Lightning Low Noise (4-step)');

const WAN_T2V = wanModel('t2v_a14b');
const WAN_I2V = wanModel('i2v_a14b');

const wanMetadata = (extra: Record<string, unknown> = {}): Record<string, unknown> => ({
  cfg_scale: 1,
  generation_mode: 'wan_t2v',
  height: 720,
  model: { base: 'wan', key: WAN_T2V.key, name: WAN_T2V.name, type: 'main' },
  negative_prompt: 'blurry',
  num_frames: 81,
  positive_prompt: 'a red fox running',
  seed: 1234,
  steps: 4,
  width: 1280,
  ...extra,
});

describe('getVideoRecallCapabilities', () => {
  it('derives per-verb availability from what the metadata contains', () => {
    expect(getVideoRecallCapabilities(wanMetadata())).toEqual({
      all: true,
      prompts: true,
      remix: true,
      seed: true,
      size: true,
    });
    expect(getVideoRecallCapabilities({ generation_mode: 'wan_t2v', seed: 7 })).toEqual({
      all: true,
      prompts: false,
      remix: false,
      seed: true,
      size: false,
    });
    // A dimension alone, or a non-positive one, is no size to recall.
    expect(getVideoRecallCapabilities({ generation_mode: 'wan_t2v', width: 1280 }).size).toBe(false);
    expect(getVideoRecallCapabilities({ generation_mode: 'wan_t2v', height: 720, width: 0 }).size).toBe(false);
  });
});

describe('parseVideoRecallRecord media', () => {
  it('suppresses the extracted first frame in extend mode', () => {
    const record = parseVideoRecallRecord({
      first_frame_image: { image_name: 'extracted.png' },
      generation_mode: 'wan_extend_video',
      last_frame_image: { image_name: 'destination.png' },
      source_video: { video_name: 'clip.mp4' },
    });

    expect(record?.media).toEqual({
      firstFrameName: null,
      lastFrameName: 'destination.png',
      sourceVideoName: 'clip.mp4',
    });
  });
});

describe('getVideoSizeRecalls', () => {
  it('inverts a preset-derived canvas exactly', () => {
    expect(getVideoSizeRecalls(WAN_T2V, 1280, 720)).toEqual([{ aspectRatioId: '16:9', targetResolution: '720p' }]);
    expect(getVideoSizeRecalls(h3Model(), 1344, 768)).toEqual([
      { aspectRatioId: '16:9', targetResolution: '768 highres' },
    ]);
  });
});

describe('deriveAcceleratorRecallState', () => {
  const settings = createDefaultVideoWidgetValues([WAN_T2V]);
  const pairEntries = [
    { isEnabled: true, model: LIGHTNING_HIGH as never, weight: 1 },
    { isEnabled: true, model: LIGHTNING_LOW as never, weight: 1 },
  ];

  it('re-enables the fast path when the recalled LoRAs are exactly the accelerator set at its steps', () => {
    expect(deriveAcceleratorRecallState(WAN_T2V, pairEntries, 4, settings)).toEqual({
      acceleratorEnabled: true,
      acceleratorLoraKeys: [LIGHTNING_HIGH.key, LIGHTNING_LOW.key],
    });
  });
});

describe('buildVideoRecallSettings', () => {
  const catalog = [WAN_T2V, WAN_I2V, h3Model(), LIGHTNING_HIGH, LIGHTNING_LOW];
  const currentValues = { ...createDefaultVideoWidgetValues([h3Model()]) };

  it('recalls only the size for the size verb, fitted to the panel model', () => {
    const wanValues = createDefaultVideoWidgetValues([WAN_T2V]);
    const metadata = wanMetadata({ height: 1280, model: { key: h3Model().key }, width: 720 });
    const result = buildVideoRecallSettings({ currentValues: wanValues, kind: 'size', metadata, models: catalog });

    expect(result?.fields).toEqual(['size']);
    // The recorded model, prompts and seed stay out; only the preset pair changes.
    expect(result?.values).toEqual({ ...wanValues, aspectRatioId: '9:16', targetResolution: '720p' });
  });

  describe('a size recall with media in the panel', () => {
    const image = (width: number, height: number) => ({ height, image_name: 'held.png', width });
    const recallSize = (values: VideoWidgetValues, width: number, height: number) =>
      buildVideoRecallSettings({
        currentValues: values,
        kind: 'size',
        metadata: wanMetadata({ height, width }),
        models: catalog,
      });
    const wanValues = createDefaultVideoWidgetValues([WAN_T2V]);

    it('writes nothing when the media would keep the frame from reaching the recorded size', () => {
      const values = { ...wanValues, firstFrameImage: image(1280, 720) };

      expect(recallSize(values, 720, 1280)).toBeNull();
      expect(resolveVideoSizeRecall(values, 720, 1280)).toMatchObject({
        effective: { height: 720, width: 1280 },
        outcome: 'media-conflict',
      });
    });
  });

  describe('a size recall that would change whether the panel can run', () => {
    const ltx2 = ltx2Model('ltx2_dev');
    const ltx2Values = createDefaultVideoWidgetValues([ltx2]);
    const audioClip = {
      clip: { duration: 4, fps: 24, height: 1024, video_name: 'song.mp4', width: 1792 },
      role: 'audio',
    };
    const metadata = wanMetadata({ height: 1024, width: 1792 });
    const recall = (values: VideoWidgetValues) =>
      buildVideoRecallSettings({ currentValues: values, kind: 'size', metadata, models: [ltx2] });

    it('writes nothing when a two-stage preset would be refused alongside the conditioning clip', () => {
      const values = {
        ...ltx2Values,
        conditioningClip: audioClip,
        targetResolution: '720p',
      } as unknown as VideoWidgetValues;

      expect(recall(values)).toBeNull();
      expect(resolveVideoSizeRecall(values, 1792, 1024)).toEqual({
        outcome: 'invalid',
        reason: expect.stringContaining('does not run a two-stage target resolution'),
      });
    });
  });

  describe('a canvas the recorded media sized, settled against them once hydrated', () => {
    const recallFirstFrame = (
      currentValues: VideoWidgetValues,
      record: Record<string, unknown>,
      hydratedFrame: VideoWidgetValues['firstFrameImage']
    ) => {
      const result = buildVideoRecallSettings({ currentValues, kind: 'all', metadata: record, models: catalog })!;

      // The pure recall restores no resolution for it: the recorded canvas lost the media's exact ratio to snapping.
      expect(result.fields).not.toContain('size');

      return settleMediaSizedRecall({ ...result, values: { ...result.values, firstFrameImage: hydratedFrame } });
    };
    const h3Values = { ...createDefaultVideoWidgetValues([h3Model()]), targetResolution: '768 lowres' as const };
    // 1800x1000 at 768 highres: 1382.4x768 is over the area cap, scaled by 0.986 to 1363.1x757.3, snapped to 1376x768.
    // Fed back, 1376x768 would give 1344x768; at 768 lowres the frame gives 768x416.
    const h3Record = {
      first_frame_image: { image_name: 'first.png' },
      generation_mode: 'minimax_h3_i2v',
      height: 768,
      model: { key: h3Model().key },
      width: 1376,
    };
    const wanValues = createDefaultVideoWidgetValues([WAN_I2V]);
    // 1000x700 at 1080p: 1542.9x1080, snapped to 1536x1088. Fed back, 1536x1088 would give 1520x1088; 720p and 480p
    // give the frame 1024x720 and 688x480.
    const wanRecord = wanMetadata({
      first_frame_image: { image_name: 'first.png' },
      generation_mode: 'wan_i2v',
      height: 1088,
      model: { key: WAN_I2V.key },
      width: 1536,
    });
    const frame = (width: number, height: number) => ({ height, image_name: 'first.png', width });

    it.each([
      ['an area-capped H3 canvas', h3Values, h3Record, frame(1800, 1000), '768 highres'],
      ['a Wan 1080p canvas snapped off its short side', wanValues, wanRecord, frame(1000, 700), '1080p'],
    ] as const)(
      'restores the target resolution of %s from the hydrated media, leaving the aspect ratio to them',
      (_, currentValues, record, hydratedFrame, targetResolution) => {
        const settled = recallFirstFrame(currentValues, record, hydratedFrame);

        expect(settled.values).toMatchObject({ aspectRatioId: currentValues.aspectRatioId, targetResolution });
        expect(settled.fields).toContain('size');
      }
    );
  });

  describe('a size more than one preset reproduces', () => {
    const h3 = h3Model();
    const h3Values = createDefaultVideoWidgetValues([h3]);
    const withFrame = (width: number, height: number): VideoWidgetValues => ({
      ...h3Values,
      firstFrameImage: { height, image_name: 'held.png', width },
    });

    it('takes the preset that fits the held frame, not just the first match', () => {
      const values = withFrame(1003, 1024);
      const lowres = { aspectRatioId: '1:1', targetResolution: '768 lowres' };

      expect(resolveVideoSizeRecall(values, 768, 768)).toEqual({ outcome: 'fits', sizeRecall: lowres });

      const result = buildVideoRecallSettings({
        currentValues: values,
        kind: 'size',
        metadata: wanMetadata({ height: 768, width: 768 }),
        models: [h3],
      });

      expect(result?.values).toEqual({ ...values, ...lowres });
      expect(getVideoDimensions(h3, result!.values)).toMatchObject({ height: 768, width: 768 });
    });
  });

  it('resolves the model by key and snaps cross-family constraints before applying values', () => {
    // Current panel is H3 (frames 124, fps 24); metadata is a Wan t2v run.
    const result = buildVideoRecallSettings({ currentValues, kind: 'all', metadata: wanMetadata(), models: catalog });

    expect(result?.values.model?.key).toBe(WAN_T2V.key);
    expect(result?.values).toMatchObject({
      aspectRatioId: '16:9',
      cfgScale: 1,
      numFrames: 81,
      seed: 1234,
      steps: 4,
      targetResolution: '720p',
    });
  });

  it('skips the model (and validates against the current one) when it is not installed', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({ model: { key: 'gone' }, num_frames: 81 }),
      models: [h3Model()],
    });

    expect(result?.fields).not.toContain('model');
    expect(result?.values.model?.base).toBe('minimax-h3');
    // 81 is off the H3 grid; it snaps rather than landing invalid.
    expect(result?.values.numFrames).toBe(90);
  });

  it('recalls LoRAs by key and restores the accelerator state that produced the video', () => {
    const metadata = wanMetadata({
      loras: [
        { model: { key: LIGHTNING_HIGH.key }, weight: 1 },
        { model: { key: LIGHTNING_LOW.key }, weight: 1 },
      ],
    });
    const result = buildVideoRecallSettings({ currentValues, kind: 'all', metadata, models: catalog });

    expect(result?.fields).toContain('loras');
    expect(result?.values.loras.map((entry) => entry.model.key)).toEqual([LIGHTNING_HIGH.key, LIGHTNING_LOW.key]);
    expect(result?.values).toMatchObject({ acceleratorEnabled: true });
  });

  it('restores reuse-primary semantics when the low-noise CFG key is absent', () => {
    const withLow = { ...createDefaultVideoWidgetValues([WAN_T2V]), cfgScaleLowNoise: 2 };
    // wan_t2v metadata without guidance_scale_low_noise: the run reused the primary CFG, so the recall must too.
    const result = buildVideoRecallSettings({
      currentValues: withLow,
      kind: 'all',
      metadata: wanMetadata(),
      models: catalog,
    });

    expect(result?.values.cfgScaleLowNoise).toBeNull();

    const withKey = buildVideoRecallSettings({
      currentValues: withLow,
      kind: 'all',
      metadata: wanMetadata({ guidance_scale_low_noise: 3 }),
      models: catalog,
    });

    expect(withKey?.values.cfgScaleLowNoise).toBe(3);

    // Below 1 the node reused the primary CFG, so the recalled state does too.
    const belowFloor = buildVideoRecallSettings({
      currentValues: withLow,
      kind: 'all',
      metadata: wanMetadata({ guidance_scale_low_noise: 0.5 }),
      models: catalog,
    });

    expect(belowFloor?.values.cfgScaleLowNoise).toBeNull();
  });
});

describe('buildVideoRecallSettings — partial records from the external recall API', () => {
  const catalog = [WAN_T2V, WAN_I2V, h3Model(), LIGHTNING_HIGH, LIGHTNING_LOW];
  const heldLora = { isEnabled: true, model: LIGHTNING_HIGH as never, weight: 0.5 };
  const holding = {
    ...createDefaultVideoWidgetValues([WAN_I2V]),
    firstFrameImage: { height: 720, image_name: 'held.png', width: 1280 },
    loras: [heldLora],
  };

  it('accepts a request without generation_mode only when told it is one', () => {
    const request = { positive_prompt: 'a heron' };

    expect(buildVideoRecallSettings({ currentValues: holding, kind: 'all', metadata: request, models: catalog })).toBe(
      null
    );
    expect(
      buildVideoRecallSettings({
        currentValues: holding,
        kind: 'all',
        metadata: request,
        models: catalog,
        requireGenerationMode: false,
      })?.values.positivePrompt
    ).toBe('a heron');
  });

  it('leaves the LoRAs, media and model a request does not name exactly as the panel has them', () => {
    const result = buildVideoRecallSettings({
      currentValues: holding,
      kind: 'all',
      metadata: { positive_prompt: 'a heron', steps: 30 },
      models: catalog,
      partial: true,
      requireGenerationMode: false,
    });

    expect(result?.fields).toEqual(['prompts', 'steps']);
    expect(result?.values).toEqual({ ...holding, positivePrompt: 'a heron', steps: 30 });
  });
});

describe('buildVideoRecallSettings — leaving the accelerator', () => {
  const catalog = [WAN_T2V, LIGHTNING_HIGH, LIGHTNING_LOW];
  const defaults = createDefaultVideoWidgetValues([WAN_T2V]);
  const accelerated = {
    ...defaults,
    acceleratorEnabled: true,
    acceleratorLoraKeys: [LIGHTNING_HIGH.key, LIGHTNING_LOW.key],
    cfgScale: 1,
    loras: [LIGHTNING_HIGH, LIGHTNING_LOW].map((model) => ({ isEnabled: true, model: model as never, weight: 1 })),
    steps: 4,
  };
  const recall = (metadata: Record<string, unknown>, partial: boolean) =>
    buildVideoRecallSettings({
      currentValues: accelerated,
      kind: 'all',
      metadata,
      models: catalog,
      partial,
      requireGenerationMode: false,
    });

  it('restores the model sampling defaults when a whole record drops the accelerator LoRAs', () => {
    expect(recall({ positive_prompt: 'p' }, false)?.values).toMatchObject({
      acceleratorEnabled: false,
      cfgScale: defaults.cfgScale,
      loras: [],
      steps: defaults.steps,
    });
    expect(defaults.steps).not.toBe(4);
  });
});

describe('ref2va reference recall', () => {
  const catalog = [WAN_T2V, WAN_I2V, h3Model(), LIGHTNING_HIGH, LIGHTNING_LOW];
  const currentValues = { ...createDefaultVideoWidgetValues([h3Model()]) };
  const h3RefMetadata = (extra: Record<string, unknown> = {}): Record<string, unknown> => ({
    generation_mode: 'minimax_h3_ref2v',
    height: 768,
    minimax_h3_references: [
      { conditioning: 'video_audio', end_frame: 47, kind: 'video', start_frame: 2, video_name: 'ref.mp4' },
      { detail: 'match', image_name: 'ref.png', kind: 'image' },
      { kind: 'video' },
    ],
    model: { base: 'minimax-h3', key: 'h3-main', name: 'MiniMax H3', type: 'main' },
    num_frames: 124,
    positive_prompt: 'a red fox',
    seed: 7,
    steps: 4,
    width: 1344,
    ...extra,
  });

  it('reports the ordered references, tolerantly skipping malformed entries', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: h3RefMetadata(),
      models: catalog,
    });

    expect(result?.fields).toContain('media');
    expect(result?.mediaNames.references).toEqual([
      { conditioning: 'video_audio', kind: 'video', name: 'ref.mp4', trim: { endFrame: 47, startFrame: 2 } },
      { detail: 'match', kind: 'image', name: 'ref.png' },
    ]);
    // A references-only record carries no frames.
    expect(result?.mediaNames.firstFrameName).toBeNull();
    expect(result?.values.references).toEqual([]);
  });

  it('recalls first/last frames beside the references of a combined run', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: h3RefMetadata({
        first_frame_image: { image_name: 'first.png' },
        generation_mode: 'minimax_h3_ref2v_frames',
        last_frame_image: { image_name: 'last.png' },
      }),
      models: catalog,
    });

    expect(result?.mediaNames).toMatchObject({ firstFrameName: 'first.png', lastFrameName: 'last.png' });
    expect(result?.mediaNames.references).toHaveLength(2);
  });
});

describe('model-position recall shapes', () => {
  const install = h3Model();
  const checkpoint: MainModelConfig = {
    base: 'minimax-h3',
    format: 'checkpoint',
    key: 'h3-ref2va-ckpt',
    name: 'MiniMax H3 Ref2VA Transformer (int8, pruned)',
    type: 'main',
    variant: 'ref2va',
  };
  const ref2vTurbo = {
    base: 'minimax-h3',
    key: 'ref2v-turbo',
    name: 'MiniMax H3 Ref2V Turbo LoRA',
    type: 'lora' as const,
  };
  const catalog = [install, checkpoint, ref2vTurbo];
  const currentValues = { ...createDefaultVideoWidgetValues([install]) };

  it('promotes a legacy transformer-override recording onto the model slot before deriving the accelerator', () => {
    // Promote the legacy transformer override before deriving Ref2V Turbo's four-step accelerator set.
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: {
        generation_mode: 'minimax_h3_ref2v',
        loras: [{ model: { key: ref2vTurbo.key }, weight: 1 }],
        minimax_h3_references: [{ detail: 'max', image_name: 'ref.png', kind: 'image' }],
        minimax_h3_transformer_model: { key: checkpoint.key },
        model: { key: install.key },
        num_frames: 124,
        steps: 4,
      },
      models: catalog,
    });

    expect(result?.values.model?.key).toBe(checkpoint.key);
    expect(result?.values.componentSourceModel?.key).toBe(install.key);
    expect(result?.values.h3TransformerModel).toBeNull();
    expect(result?.values.modelKey).toBe(checkpoint.key);
    expect(result?.values).toMatchObject({ acceleratorEnabled: true, acceleratorLoraKeys: [ref2vTurbo.key] });
  });

  it('preserves an explicitly matching Ref2VA hybrid through partial recall and catalog sync', () => {
    const hybrid = { ...checkpoint, key: 'quality-base', variant: 'fl2va' };
    const models = [...catalog, hybrid];
    const holding: VideoWidgetValues = {
      ...currentValues,
      model: checkpoint,
      modelKey: checkpoint.key,
      h3TaskOverride: 'ref2va',
      h3HybridBaseModel: hybrid,
      h3HybridStartBlock: 30,
    };
    const result = buildVideoRecallSettings({
      currentValues: holding,
      kind: 'seed',
      metadata: { generation_mode: 'minimax_h3_ref2v', seed: 123 },
      models,
    });
    expect(result).not.toBeNull();
    const synced = syncVideoWidgetValuesWithModels(result!.values, models);
    expect(synced).toMatchObject({
      seed: 123,
      h3TaskOverride: 'ref2va',
      h3HybridBaseModel: hybrid,
      h3HybridStartBlock: 30,
    });
  });
});

describe('portable records (metadata_version 1.0.0)', () => {
  const catalog = [WAN_T2V, WAN_I2V, h3Model(), LIGHTNING_HIGH, LIGHTNING_LOW];
  const currentValues = { ...createDefaultVideoWidgetValues([h3Model()]) };

  it('reads the canonical low-noise CFG key and its pre-1.0 alias alike', () => {
    const withLow = { ...createDefaultVideoWidgetValues([WAN_T2V]), cfgScaleLowNoise: 2 };
    const canonical = buildVideoRecallSettings({
      currentValues: withLow,
      kind: 'all',
      metadata: wanMetadata({ wan_guidance_scale_low_noise: 3.5 }),
      models: catalog,
    });
    const legacy = buildVideoRecallSettings({
      currentValues: withLow,
      kind: 'all',
      metadata: wanMetadata({ guidance_scale_low_noise: 3.5 }),
      models: catalog,
    });

    expect(canonical?.values.cfgScaleLowNoise).toBe(3.5);
    expect(legacy?.values.cfgScaleLowNoise).toBe(3.5);
  });

  it('falls back to name, base and type when neither key nor hash matches', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({
        model: { base: 'wan', hash: 'blake3:nope', key: 'foreign-key', name: WAN_T2V.name, type: 'main' },
      }),
      models: catalog,
    });

    expect(result?.values.model?.key).toBe(WAN_T2V.key);

    const wrongType = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({ model: { base: 'wan', key: 'foreign-key', name: WAN_T2V.name, type: 'lora' } }),
      models: catalog,
    });

    expect(wrongType?.fields).not.toContain('model');
  });

  it('resolves LoRAs and Wan components through the same ladder', () => {
    const hashedLora = { ...LIGHTNING_HIGH, hash: 'blake3:high' };
    const umt5 = { base: 'any', hash: 'blake3:umt5', key: 'local-umt5', name: 'UMT5-XXL', type: 'wan_t5_encoder' };
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({
        loras: [{ model: { hash: 'blake3:high', key: 'foreign-lora' }, weight: 0.8 }],
        // The canonical key for the standalone encoder; a record may also spell it `wan_t5_encoder`.
        wan_t5_encoder_model: {
          base: 'any',
          hash: 'blake3:umt5',
          key: 'foreign-umt5',
          name: 'UMT5-XXL',
          type: 'wan_t5_encoder',
        },
      }),
      models: [WAN_T2V, hashedLora, umt5],
    });

    expect(result?.values.loras).toEqual([{ isEnabled: true, model: hashedLora, weight: 0.8 }]);
    expect(result?.values.wanT5EncoderModel).toEqual(umt5);

    const lowExpert = wanModel('t2v_a14b', 'checkpoint', 'low-expert');
    const aliased = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({ transformer_low_noise: { key: lowExpert.key }, wan_t5_encoder: { key: 'local-umt5' } }),
      models: [WAN_T2V, umt5, lowExpert],
    });

    expect(aliased?.values.wanT5EncoderModel).toEqual(umt5);
    expect(aliased?.values.wanLowNoiseModel).toEqual(lowExpert);
  });

  it('reads a pre-identifier model record, which may omit the type', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({ model: { base_model: 'wan', model_name: WAN_T2V.name } }),
      models: catalog,
    });

    expect(result?.values.model?.key).toBe(WAN_T2V.key);
  });
});

const ltx2Model = (variant: string, format = 'checkpoint', key = `ltx2-${variant}`): MainModelConfig => ({
  base: 'ltx-2',
  format,
  key,
  name: `LTX-2 ${variant}`,
  type: 'main',
  variant,
});

const LTX2_DEV = ltx2Model('ltx2_dev');
const LTX2_DISTILLED = ltx2Model('ltx2_distilled');
const LTX2_COMPONENTS = ltx2Model('ltx2_dev', 'diffusers', 'ltx2-components');
const LTX2_ENCODER: GenerationModelCatalogItem = {
  base: 'ltx-2',
  key: 'gemma4',
  name: 'LTX-2.5 Text Encoder',
  type: 'gemma4_encoder',
};

const ltx2Metadata = (extra: Record<string, unknown> = {}): Record<string, unknown> => ({
  cfg_scale: 3,
  fps: 24,
  generation_mode: 'ltx2_t2v',
  height: 704,
  ltx2_audio_cfg_scale: 7,
  ltx2_component_source: LTX2_COMPONENTS,
  ltx2_modality_scale: 3,
  ltx2_stg_scale: 1,
  ltx2_text_encoder_model: LTX2_ENCODER,
  model: LTX2_DEV,
  negative_prompt: 'blurry',
  num_frames: 121,
  positive_prompt: 'a ginger cat',
  seed: 99,
  steps: 30,
  width: 1248,
  ...extra,
});

describe('LTX-2 recall', () => {
  const catalog = [LTX2_DEV, LTX2_DISTILLED, LTX2_COMPONENTS, LTX2_ENCODER, WAN_T2V];
  const currentValues = createDefaultVideoWidgetValues([WAN_T2V]);

  it('turns auto duration off when it restores a length, so the recalled clip runs at that length', () => {
    // Same model, head installed and selected: model selection keeps both, so only recall's own rule can turn auto
    // duration off. Left on, the recalled 121 frames would be a ceiling, not the length.
    const head: GenerationModelCatalogItem = {
      base: 'ltx-2',
      key: 'duration-head',
      name: 'LTX-2.5 Duration Head',
      type: 'ltx2_duration_head',
    };
    const result = buildVideoRecallSettings({
      currentValues: { ...createDefaultVideoWidgetValues([LTX2_DEV]), autoDuration: true, ltx2DurationHeadModel: head },
      kind: 'all',
      metadata: ltx2Metadata({ num_frames: 121 }),
      models: [...catalog, head],
    });

    expect(result?.fields).toContain('frames');
    expect(result?.values).toMatchObject({ autoDuration: false, ltx2DurationHeadModel: head, numFrames: 121 });
  });

  it('recalls a two-stage run as the preset that produces its canvas', () => {
    // A two-stage run records its final canvas, and no single-stage preset resolves to it -- so the preset comes back
    // from the size alone, without the metadata having to name the stage count.
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: ltx2Metadata({
        height: 1024,
        ltx2_base_height: 512,
        ltx2_base_width: 896,
        ltx2_two_stage: true,
        width: 1792,
      }),
      models: catalog,
    });

    expect(result?.values.targetResolution).toBe('1024p');
    expect(result?.values.aspectRatioId).toBe('16:9');
  });
});

it('reads the legacy Wan VAE slot', () => {
  const vae = { key: 'legacy-vae', base: 'wan', name: 'Wan VAE', type: 'vae' };
  expect(parseVideoRecallRecord({ generation_mode: 'wan_t2v', wan_vae_model: vae })?.components).toContainEqual({
    modelType: 'vae',
    valuesKey: 'vae',
    ref: { key: vae.key, base: vae.base, name: vae.name, type: vae.type, hash: null },
  });
});

describe('H3 overridden task recall', () => {
  it.each(['fl2va', 'ref2va'] as const)('round-trips a %s override with its Turbo through graph metadata', (task) => {
    const model = { ...h3Model('hybrid'), format: 'checkpoint', variant: task === 'ref2va' ? 'fl2va' : 'ref2va' };
    const component = h3Model('components');
    const flTurbo = lora('MiniMax H3 Turbo LoRA', 'minimax-h3', null);
    const refTurbo = lora('MiniMax H3 Ref2V Turbo LoRA', 'minimax-h3', null);
    const catalog = [model, component, flTurbo, refTurbo];
    const references = [
      { kind: 'image' as const, detail: 'match' as const, image: { image_name: 'ref.png', width: 512, height: 512 } },
    ];
    const settings = getAcceleratorToggleResult(
      {
        ...getDefaultVideoSettings(model, catalog),
        h3TaskOverride: task,
        references: task === 'ref2va' ? references : [],
      },
      model,
      catalog,
      true
    ).settings;
    const selectedTurbo = task === 'ref2va' ? refTurbo : flTurbo;
    expect(settings.acceleratorLoraKeys).toEqual([selectedTurbo.key]);
    const graph = compileVideoGraph(settings, model).backendGraph;
    expect(graph.nodes.model_loader).toMatchObject({ task_variant_override: task });
    const metadata = recordOf(compileVideoGraph(settings, model), settings);
    const result = buildVideoRecallSettings({
      currentValues: createDefaultVideoWidgetValues(catalog),
      kind: 'all',
      metadata,
      models: catalog,
    })!;
    expect(result.values.h3TaskOverride).toBe(task);
    expect(result.values.model).toBe(model);
    expect(result.values.acceleratorEnabled).toBe(true);
    expect(result.values.acceleratorLoraKeys).toEqual([selectedTurbo.key]);
    // Hydration supplies the media returned separately by recall.
    const restored = syncVideoWidgetValuesWithModels({ ...result.values, references: settings.references }, catalog);
    expect(compileVideoGraph(restored, restored.model!).backendGraph.nodes.model_loader).toMatchObject({
      task_variant_override: task,
    });
    expect(restored.steps).toBe(settings.steps);
  });

  it('restores the effective task from generation_mode for the same checkpoint', () => {
    const model = { ...h3Model('hybrid'), format: 'checkpoint', variant: 'fl2va' };
    const values = createDefaultVideoWidgetValues([model]);
    const result = buildVideoRecallSettings({
      currentValues: values,
      kind: 'all',
      metadata: {
        generation_mode: 'minimax_h3_ref2v',
        model: { base: 'minimax-h3', key: model.key, name: model.name, type: 'main' },
      },
      models: [model],
    });
    expect(result?.values.h3TaskOverride).toBe('ref2va');
    expect(result?.values.model).toBe(model);
    expect(model.variant).toBe('fl2va');
  });

  it('does not apply the recorded task to an unrelated model when the recorded model is unavailable', () => {
    const current = { ...h3Model('current'), format: 'checkpoint', variant: 'fl2va' };
    const values = createDefaultVideoWidgetValues([current]);
    const result = buildVideoRecallSettings({
      currentValues: values,
      kind: 'all',
      metadata: {
        generation_mode: 'minimax_h3_ref2v',
        positive_prompt: 'test scene',
        model: { base: 'minimax-h3', key: 'uninstalled', name: 'Missing model', type: 'main' },
      },
      models: [current],
    });
    expect(result?.values.h3TaskOverride).toBeUndefined();
    expect(result?.values.model?.variant).toBe('fl2va');
  });
});

describe('legacy H3 duration recall', () => {
  const model = h3Model();
  const currentValues = { ...createDefaultVideoWidgetValues([model]), numFrames: 175 };
  const metadata = (duration: unknown, extra: Record<string, unknown> = {}) => ({
    generation_mode: 'minimax_h3_t2v',
    model: { key: model.key },
    minimax_h3_duration_seconds: duration,
    ...extra,
  });

  it.each([
    [5, 124],
    [6, 141],
    [7, 175],
    [8, 192],
    [9, 209],
    [10, 243],
    [11, 260],
    [12, 294],
    [13, 311],
    [14, 345],
  ])('restores the emitted frame count for the legacy %is duration', (duration, expectedFrames) => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: metadata(duration),
      models: [model],
    });
    expect(result?.fields).toContain('frames');
    expect(result?.values).toMatchObject({ numFrames: expectedFrames, fps: 24 });
  });
});

describe('H3 draft size recall', () => {
  it.each([
    [864, 480, '16:9', '480p'],
    [960, 544, '16:9', '544p'],
    [544, 960, '9:16', '544p'],
    [480, 480, '1:1', '480p'],
  ] as const)('recalls %sx%s as %s %s', (width, height, aspectRatioId, targetResolution) => {
    expect(getVideoSizeRecalls(h3Model(), width, height)).toContainEqual({ aspectRatioId, targetResolution });
  });
});

describe('equivalent H3 media-sized presets', () => {
  const model = h3Model();
  const frame = { width: 1000, height: 700, image_name: 'legacy-frame.png' };

  it.each(['768 highres', '768 lowres', '544p'] as const)(
    'recalls the original 768x544 canvas from %s without increasing its size',
    (targetResolution) => {
      const values = { ...createDefaultVideoWidgetValues([model]), firstFrameImage: frame, targetResolution };
      const result = buildVideoRecallSettings({
        currentValues: values,
        kind: 'all',
        metadata: {
          generation_mode: 'minimax_h3_i2v',
          width: 768,
          height: 544,
          first_frame_image: { image_name: frame.image_name },
        },
        models: [model],
      });
      expect(result).not.toBeNull();
      const settled = settleMediaSizedRecall({ ...result!, values: { ...result!.values, firstFrameImage: frame } });
      expect(settled.values.targetResolution).toBe(targetResolution === '544p' ? '544p' : '768 lowres');
      expect(getVideoDimensions(model, settled.values)).toMatchObject({ width: 768, height: 544 });
      const sizeOnly = resolveVideoSizeRecall(values, 768, 544);
      expect(sizeOnly).toEqual({
        outcome: 'fits',
        sizeRecall: { targetResolution: targetResolution === '544p' ? '544p' : '768 lowres' },
      });
    }
  );
});

describe('Wan LoRA expert target recall', () => {
  const highLora = { ...LIGHTNING_HIGH, key: 'wan-expert-high' };
  const lowLora = { ...LIGHTNING_LOW, key: 'wan-expert-low' };
  const dualLora = { base: 'wan', key: 'wan-expert-dual', name: 'Wan Dual', type: 'lora', variant: 'a14b' };
  const currentValues = createDefaultVideoWidgetValues([WAN_T2V]);

  it('restores target and low_noise_weight without altering untargeted LoRA records', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({
        loras: [
          { model: { key: highLora.key }, target: 'high', weight: 0.9 },
          { model: { key: lowLora.key }, target: 'low', weight: 0.5 },
          { low_noise_weight: 0.3, model: { key: dualLora.key }, target: 'both', weight: 0.8 },
          { model: { key: LIGHTNING_HIGH.key }, weight: 1 },
        ],
      }),
      models: [WAN_T2V, highLora, lowLora, dualLora, LIGHTNING_HIGH],
    });

    expect(result?.values.loras).toEqual([
      { isEnabled: true, model: highLora, target: 'high', weight: 0.9 },
      { isEnabled: true, model: lowLora, target: 'low', weight: 0.5 },
      { isEnabled: true, lowNoiseWeight: 0.3, model: dualLora, target: 'both', weight: 0.8 },
      { isEnabled: true, model: LIGHTNING_HIGH, weight: 1 },
    ]);
  });

  it('folds legacy dual high/low entries for the same LoRA into one both entry', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({
        loras: [
          { model: { key: dualLora.key }, target: 'low', weight: 0.4 },
          { model: { key: dualLora.key }, target: 'high', weight: 0.85 },
        ],
      }),
      models: [WAN_T2V, dualLora],
    });

    expect(result?.values.loras).toEqual([
      { isEnabled: true, lowNoiseWeight: 0.4, model: dualLora, target: 'both', weight: 0.85 },
    ]);
  });

  it('deduplicates non-opposing duplicate entries for the same LoRA key', () => {
    const sameTarget = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({
        loras: [
          { model: { key: highLora.key }, target: 'high', weight: 0.9 },
          { model: { key: highLora.key }, target: 'high', weight: 0.5 },
          { model: { key: dualLora.key }, target: 'high', weight: 0.8 },
          { model: { key: dualLora.key }, target: 'low', weight: 0.4 },
          { model: { key: dualLora.key }, target: 'low', weight: 0.2 },
          { model: { key: LIGHTNING_HIGH.key }, weight: 1 },
          { model: { key: LIGHTNING_HIGH.key }, weight: 0.6 },
        ],
      }),
      models: [WAN_T2V, highLora, dualLora, LIGHTNING_HIGH],
    });

    expect(sameTarget?.values.loras).toEqual([
      { isEnabled: true, model: highLora, target: 'high', weight: 0.9 },
      { isEnabled: true, lowNoiseWeight: 0.4, model: dualLora, target: 'both', weight: 0.8 },
      { isEnabled: true, model: LIGHTNING_HIGH, weight: 1 },
    ]);
  });
});
