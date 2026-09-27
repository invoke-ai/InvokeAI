import type { GenerationModelCatalogItem, MainModelConfig } from '@features/generation/contracts';

import { createDefaultVideoWidgetValues } from '@features/video';
import { getVideoPromptPolicy } from '@features/video/core/videoPolicies';
import { describe, expect, it } from 'vitest';

import {
  buildVideoRecallSettings,
  deriveAcceleratorRecallState,
  EMPTY_VIDEO_RECALL_CAPABILITIES,
  getRecallableMediaNames,
  getVideoRecallCapabilities,
  getVideoSizeRecall,
  isVideoGenerationMetadata,
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

describe('isVideoGenerationMetadata', () => {
  it('accepts only the video generation modes', () => {
    expect(isVideoGenerationMetadata(wanMetadata())).toBe(true);
    expect(isVideoGenerationMetadata({ generation_mode: 'minimax_h3_flf2v' })).toBe(true);
    // Image metadata — including Wan still-image modes — is not video metadata.
    expect(isVideoGenerationMetadata({ generation_mode: 'wan_txt2img' })).toBe(false);
    expect(isVideoGenerationMetadata({ generation_mode: 'sdxl_txt2img' })).toBe(false);
    expect(isVideoGenerationMetadata({})).toBe(false);
    expect(isVideoGenerationMetadata(null)).toBe(false);
  });
});

describe('getVideoRecallCapabilities', () => {
  it('is empty for non-video metadata', () => {
    expect(getVideoRecallCapabilities({ generation_mode: 'sdxl_txt2img', seed: 5 })).toEqual(
      EMPTY_VIDEO_RECALL_CAPABILITIES
    );
  });

  it('derives per-verb availability from what the metadata contains', () => {
    expect(getVideoRecallCapabilities(wanMetadata())).toEqual({ all: true, prompts: true, remix: true, seed: true });
    expect(getVideoRecallCapabilities({ generation_mode: 'wan_t2v', seed: 7 })).toEqual({
      all: true,
      prompts: false,
      remix: false,
      seed: true,
    });
  });
});

describe('getRecallableMediaNames', () => {
  it('suppresses the extracted first frame in extend mode', () => {
    const names = getRecallableMediaNames({
      first_frame_image: { image_name: 'extracted.png' },
      generation_mode: 'wan_extend_video',
      last_frame_image: { image_name: 'destination.png' },
      source_video: { video_name: 'clip.mp4' },
    });

    expect(names).toEqual({ firstFrameName: null, lastFrameName: 'destination.png', sourceVideoName: 'clip.mp4' });
  });

  it('keeps user keyframes when no source video was recorded', () => {
    expect(
      getRecallableMediaNames({ first_frame_image: { image_name: 'first.png' }, generation_mode: 'wan_i2v' })
    ).toEqual({ firstFrameName: 'first.png', lastFrameName: null, sourceVideoName: null });
  });
});

describe('getVideoSizeRecall', () => {
  it('inverts a preset-derived canvas exactly', () => {
    expect(getVideoSizeRecall(WAN_T2V, 1280, 720)).toEqual({ aspectRatioId: '16:9', targetResolution: '720p' });
    expect(getVideoSizeRecall(h3Model(), 1344, 768)).toEqual({
      aspectRatioId: '16:9',
      targetResolution: '768 highres',
    });
  });

  it('rejects sizes that came from conditioning media rather than a preset', () => {
    // 1920x1080-shaped media at some odd snap no preset reproduces.
    expect(getVideoSizeRecall(WAN_T2V, 848, 464)).toBeNull();
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

  it('reads the step count off the recalled LoRAs, not off what the panel is running', () => {
    // A panel already running an 8-step Turbo LoRA must not shift the step
    // count a 6-step one is checked against.
    const turbo = { base: 'minimax-h3', key: 'turbo', name: 'MiniMax H3 Turbo LoRA', type: 'lora' as const };
    const lightx2v = {
      base: 'minimax-h3',
      key: 'lightx2v',
      name: 'MiniMax H3 LightX2V Turbo LoRA',
      type: 'lora' as const,
    };
    const model = h3Model();
    const running = {
      ...createDefaultVideoWidgetValues([model, lightx2v as never]),
      acceleratorEnabled: true,
      acceleratorLoraKeys: [lightx2v.key],
      loras: [{ isEnabled: true, model: lightx2v as never, weight: 1 }],
      steps: 8,
    };
    const recalled = [{ isEnabled: true, model: turbo as never, weight: 1 }];

    expect(deriveAcceleratorRecallState(model, recalled, 6, running)).toEqual({
      acceleratorEnabled: true,
      acceleratorLoraKeys: [turbo.key],
    });
  });

  it('stays off at non-accelerator steps or with a partial pair', () => {
    expect(deriveAcceleratorRecallState(WAN_T2V, pairEntries, 40, settings)).toEqual({
      acceleratorEnabled: false,
      acceleratorLoraKeys: [],
    });
    expect(deriveAcceleratorRecallState(WAN_T2V, [pairEntries[0]!], 4, settings)).toEqual({
      acceleratorEnabled: false,
      acceleratorLoraKeys: [],
    });
  });
});

describe('buildVideoRecallSettings', () => {
  const catalog = [WAN_T2V, WAN_I2V, h3Model(), LIGHTNING_HIGH, LIGHTNING_LOW];
  const currentValues = { ...createDefaultVideoWidgetValues([h3Model()]) };

  it('recalls prompts into the video widget values, leaving everything else alone', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'prompts',
      metadata: wanMetadata(),
      models: catalog,
    });

    expect(result?.fields).toEqual(['prompts']);
    expect(result?.values).toEqual({
      ...currentValues,
      negativePrompt: 'blurry',
      negativePromptEnabled: true,
      positivePrompt: 'a red fox running',
    });
  });

  it('recalls only the seed for the seed verb', () => {
    const result = buildVideoRecallSettings({ currentValues, kind: 'seed', metadata: wanMetadata(), models: catalog });

    expect(result?.fields).toEqual(['seed']);
    expect(result?.values).toMatchObject({ seed: 1234, seedMode: 'fixed' });
  });

  it('remix recalls everything except the seed', () => {
    const result = buildVideoRecallSettings({ currentValues, kind: 'remix', metadata: wanMetadata(), models: catalog });

    expect(result?.fields).not.toContain('seed');
    expect(result?.values.seedMode).toBe(currentValues.seedMode);
    expect(result?.fields).toEqual(expect.arrayContaining(['model', 'frames', 'steps', 'cfg', 'size']));
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

  it('clears existing LoRAs when the video ran without any', () => {
    const withLora = {
      ...currentValues,
      loras: [{ isEnabled: true, model: LIGHTNING_HIGH as never, weight: 1 }],
    };
    const result = buildVideoRecallSettings({
      currentValues: withLora,
      kind: 'all',
      metadata: wanMetadata(),
      models: catalog,
    });

    expect(result?.fields).toContain('loras');
    expect(result?.values.loras).toEqual([]);
    expect(result?.values.acceleratorEnabled).toBe(false);
  });

  it('reports recallable media names but leaves hydration to the executor', () => {
    const metadata = wanMetadata({
      first_frame_image: { image_name: 'first.png' },
      generation_mode: 'wan_i2v',
      model: { key: WAN_I2V.key },
    });
    const result = buildVideoRecallSettings({ currentValues, kind: 'all', metadata, models: catalog });

    expect(result?.fields).toContain('media');
    expect(result?.mediaNames).toEqual({
      conditioningClip: null,
      firstFrameName: 'first.png',
      lastFrameName: null,
      references: [],
      sourceVideoName: null,
      sourceVideoTrim: null,
    });
    expect(result?.values.firstFrameImage).toBeNull();
  });

  it('recalls the recorded Wan fps and carries the source-clip trim bounds for the executor', () => {
    const withFps = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({ fps: 24 }),
      models: catalog,
    });

    expect(withFps?.fields).toContain('fps');
    expect(withFps?.values.fps).toBe(24);

    const extend = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({
        generation_mode: 'wan_extend_video',
        model: { key: WAN_I2V.key },
        source_video: { video_name: 'clip.mp4' },
        source_video_end_frame: 50,
        source_video_start_frame: 10,
      }),
      models: catalog,
    });

    expect(extend?.mediaNames.sourceVideoName).toBe('clip.mp4');
    expect(extend?.mediaNames.sourceVideoTrim).toEqual({ endFrame: 50, startFrame: 10 });
  });

  it('restores reuse-primary semantics when the low-noise CFG key is absent', () => {
    const withLow = { ...createDefaultVideoWidgetValues([WAN_T2V]), cfgScaleLowNoise: 2 };
    // wan_t2v metadata without guidance_scale_low_noise: the run reused the
    // primary CFG, so the recalled state must too.
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

  it('clears held conditioning media when the recorded run had none — mode is part of the recall', () => {
    const withFrame = {
      ...createDefaultVideoWidgetValues([WAN_I2V]),
      firstFrameImage: { height: 720, image_name: 'held.png', width: 1280 },
    };
    const result = buildVideoRecallSettings({
      currentValues: withFrame,
      kind: 'all',
      metadata: wanMetadata(),
      models: catalog,
    });

    expect(result?.fields).toContain('media');
    expect(result?.values.firstFrameImage).toBeNull();
    expect(result?.values.sourceVideo).toBeNull();
  });

  it('does not flip the negative toggle on for an empty recorded negative', () => {
    const empty = buildVideoRecallSettings({
      currentValues,
      kind: 'prompts',
      metadata: wanMetadata({ negative_prompt: '' }),
      models: catalog,
    });

    expect(empty?.values).toEqual({ ...currentValues, positivePrompt: 'a red fox running' });

    const explicitNull = buildVideoRecallSettings({
      currentValues,
      kind: 'prompts',
      metadata: wanMetadata({ negative_prompt: null }),
      models: catalog,
    });

    expect(explicitNull?.values).toMatchObject({ negativePrompt: '', negativePromptEnabled: false });
  });

  it('clears current LoRAs when the recorded ones are no longer installed', () => {
    const withLora = {
      ...currentValues,
      loras: [{ isEnabled: true, model: LIGHTNING_HIGH as never, weight: 1 }],
    };
    const result = buildVideoRecallSettings({
      currentValues: withLora,
      kind: 'all',
      metadata: wanMetadata({ loras: [{ model: { key: 'uninstalled' }, weight: 1 }] }),
      models: [WAN_T2V, LIGHTNING_HIGH],
    });

    expect(result?.values.loras).toEqual([]);
    expect(result?.fields).toContain('loras');
  });

  it('returns null for non-video metadata', () => {
    expect(
      buildVideoRecallSettings({
        currentValues,
        kind: 'all',
        metadata: { generation_mode: 'sdxl_txt2img', seed: 3 },
        models: catalog,
      })
    ).toBeNull();
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

  it('replaces the LoRA set only when the request names one, an empty list included', () => {
    const cleared = buildVideoRecallSettings({
      currentValues: holding,
      kind: 'all',
      metadata: { loras: [] },
      models: catalog,
      partial: true,
      requireGenerationMode: false,
    });

    expect(cleared?.fields).toEqual(['loras']);
    expect(cleared?.values.loras).toEqual([]);
  });

  it('names a media slot without clearing the others; hydration displaces only its rivals', () => {
    const result = buildVideoRecallSettings({
      currentValues: holding,
      kind: 'all',
      metadata: { last_frame_image: { image_name: 'last.png' } },
      models: catalog,
      partial: true,
      requireGenerationMode: false,
    });

    expect(result?.mediaNames).toMatchObject({ firstFrameName: null, lastFrameName: 'last.png' });
    expect(result?.values.firstFrameImage).toEqual(holding.firstFrameImage);
  });

  it('clears the references for an explicitly empty list, and only then', () => {
    const panel = {
      ...createDefaultVideoWidgetValues([h3Model()]),
      references: [
        { detail: 'max' as const, image: { height: 512, image_name: 'ref.png', width: 512 }, kind: 'image' as const },
      ],
    };
    const build = (metadata: Record<string, unknown>) =>
      buildVideoRecallSettings({
        currentValues: panel,
        kind: 'all',
        metadata,
        models: catalog,
        partial: true,
        requireGenerationMode: false,
      });

    expect(build({ minimax_h3_references: [] })).toMatchObject({ fields: ['media'], values: { references: [] } });
    expect(build({ positive_prompt: 'p' })?.values.references).toEqual(panel.references);
  });

  it('keeps the panel hybrid base and may set its start block', () => {
    const fl2vaBase: MainModelConfig = { ...h3Model('h3-fl2va-base'), format: 'checkpoint' };
    const panel = {
      ...createDefaultVideoWidgetValues([h3Model()]),
      h3HybridBaseModel: fl2vaBase,
      h3HybridStartBlock: 25,
    };
    const result = buildVideoRecallSettings({
      currentValues: panel,
      kind: 'all',
      metadata: { minimax_h3_hybrid_start_block: 30 },
      models: [...catalog, fl2vaBase],
      partial: true,
      requireGenerationMode: false,
    });

    expect(result?.values.h3HybridBaseModel).toBe(fl2vaBase);
    expect(result?.values.h3HybridStartBlock).toBe(30);
  });

  it('reads an explicit null negative prompt as turning it off', () => {
    const result = buildVideoRecallSettings({
      currentValues: { ...holding, negativePrompt: 'blurry', negativePromptEnabled: true },
      kind: 'all',
      metadata: { negative_prompt: null },
      models: catalog,
      partial: true,
      requireGenerationMode: false,
    });

    expect(result?.values).toMatchObject({ negativePrompt: '', negativePromptEnabled: false });
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

  it('keeps the sampling values the record names', () => {
    expect(recall({ loras: [], steps: 12 }, true)?.values).toMatchObject({
      acceleratorEnabled: false,
      cfgScale: defaults.cfgScale,
      steps: 12,
    });
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

  it('recognizes the ref2v generation mode as video metadata', () => {
    expect(isVideoGenerationMetadata({ generation_mode: 'minimax_h3_ref2v' })).toBe(true);
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
    // References replace the frame/source slots.
    expect(result?.mediaNames.firstFrameName).toBeNull();
    expect(result?.values.references).toEqual([]);
  });

  it("clears the panel's held media so the executor re-hydrates references on a clean slate", () => {
    const holding = {
      ...currentValues,
      firstFrameImage: { height: 4, image_name: 'held.png', width: 4 },
    };
    const result = buildVideoRecallSettings({
      currentValues: holding,
      kind: 'all',
      metadata: h3RefMetadata(),
      models: catalog,
    });

    expect(result?.values.firstFrameImage).toBeNull();
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

  it('recalls the hybrid quality base together with its start block, never the block alone', () => {
    const fl2vaBase: MainModelConfig = {
      base: 'minimax-h3',
      format: 'checkpoint',
      key: 'h3-fl2va-ckpt',
      name: 'MiniMax H3 FL2VA Transformer (int8, pruned)',
      type: 'main',
      variant: 'fl2va',
    };
    const metadata = {
      generation_mode: 'minimax_h3_ref2v',
      minimax_h3_component_source: { key: install.key },
      minimax_h3_hybrid_base_model: { key: fl2vaBase.key },
      minimax_h3_hybrid_start_block: 30,
      minimax_h3_references: [{ detail: 'max', image_name: 'ref.png', kind: 'image' }],
      model: { key: checkpoint.key },
      num_frames: 124,
    };
    const result = buildVideoRecallSettings({ currentValues, kind: 'all', metadata, models: [...catalog, fl2vaBase] });

    expect(result?.fields).toContain('components');
    expect(result?.values.model?.key).toBe(checkpoint.key);
    expect(result?.values.h3HybridBaseModel?.key).toBe(fl2vaBase.key);
    expect(result?.values.h3HybridStartBlock).toBe(30);

    // With the base uninstalled, the block stays at the panel's value: a start block only
    // means something for the base it was recorded with.
    const gone = buildVideoRecallSettings({ currentValues, kind: 'all', metadata, models: catalog });

    expect(gone?.values.h3HybridBaseModel).toBeNull();
    expect(gone?.values.h3HybridStartBlock).toBe(currentValues.h3HybridStartBlock);

    // An uninstalled recorded hybrid base must not reuse the panel's base or receive its recorded blocks.
    const otherBase: MainModelConfig = { ...fl2vaBase, key: 'h3-fl2va-other', name: 'Another FL2VA' };
    const holding = { ...currentValues, h3HybridBaseModel: otherBase, h3HybridStartBlock: 12 };
    const onto = buildVideoRecallSettings({
      currentValues: holding,
      kind: 'all',
      metadata,
      models: [...catalog, otherBase],
    });

    expect(onto?.fields).toContain('components');
    expect(onto?.values.h3HybridBaseModel).toBeNull();
    expect(onto?.values.h3HybridStartBlock).toBe(12);
  });

  it.each(['all', 'remix'] as const)(
    'clears the hybrid quality base on a %s recall of a run recorded without it',
    (kind) => {
      const fl2vaBase: MainModelConfig = {
        base: 'minimax-h3',
        format: 'checkpoint',
        key: 'h3-fl2va-ckpt',
        name: 'MiniMax H3 FL2VA Transformer (int8, pruned)',
        type: 'main',
        variant: 'fl2va',
      };
      // No component source recorded either, so the cleared base is the only component change.
      const metadata = {
        generation_mode: 'minimax_h3_ref2v',
        minimax_h3_references: [{ detail: 'max', image_name: 'ref.png', kind: 'image' }],
        model: { key: checkpoint.key },
        num_frames: 124,
      };
      const holding = { ...currentValues, h3HybridBaseModel: fl2vaBase, h3HybridStartBlock: 12 };
      const result = buildVideoRecallSettings({
        currentValues: holding,
        kind,
        metadata,
        models: [...catalog, fl2vaBase],
      });

      expect(result?.fields).toContain('components');
      expect(result?.values.model?.key).toBe(checkpoint.key);
      expect(result?.values.h3HybridBaseModel).toBeNull();
      // The block is hidden without a base and only means something with one; it is left alone.
      expect(result?.values.h3HybridStartBlock).toBe(12);

      // A panel without a base has nothing to clear, and the toast must not claim a component change.
      const bare = buildVideoRecallSettings({ currentValues, kind, metadata, models: [...catalog, fl2vaBase] });

      expect(bare?.values.h3HybridBaseModel).toBeNull();
      expect(bare?.fields).not.toContain('components');

      // Report a hybrid base cleared by the model transition as cleared by recall.
      const toFl2va = buildVideoRecallSettings({
        currentValues: holding,
        kind,
        metadata: {
          ...metadata,
          generation_mode: 'minimax_h3_t2v',
          minimax_h3_references: undefined,
          model: { key: fl2vaBase.key },
        },
        models: [...catalog, fl2vaBase],
      });

      expect(toFl2va?.values.model?.key).toBe(fl2vaBase.key);
      expect(toFl2va?.values.h3HybridBaseModel).toBeNull();
      expect(toFl2va?.fields).toContain('components');

      // A seed recall never reaches the components: the held base survives it.
      const seedOnly = buildVideoRecallSettings({
        currentValues: holding,
        kind: 'seed',
        metadata: { ...metadata, seed: 7 },
        models: [...catalog, fl2vaBase],
      });

      expect(seedOnly?.fields).toEqual(['seed']);
      expect(seedOnly?.values.h3HybridBaseModel).toEqual(fl2vaBase);
    }
  );

  it('recalls the recorded component source for a checkpoint-main recording', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: {
        generation_mode: 'minimax_h3_ref2v',
        minimax_h3_component_source: { key: install.key },
        minimax_h3_references: [{ detail: 'max', image_name: 'ref.png', kind: 'image' }],
        model: { key: checkpoint.key },
        num_frames: 124,
      },
      models: catalog,
    });

    expect(result?.fields).toContain('model');
    expect(result?.values.model?.key).toBe(checkpoint.key);
    expect(result?.values.componentSourceModel?.key).toBe(install.key);
  });
  it('promotes the recorded transformer even when the recorded install itself is gone', () => {
    // Promote the recorded transformer even when the missing install falls back to the panel's checkpoint.
    const panelCheckpoint: MainModelConfig = {
      base: 'minimax-h3',
      format: 'checkpoint',
      key: 'h3-fl2va-ckpt',
      name: 'MiniMax H3 FL2VA Transformer (int8)',
      type: 'main',
      variant: 'fl2va',
    };
    const result = buildVideoRecallSettings({
      currentValues: { ...currentValues, model: panelCheckpoint, modelKey: panelCheckpoint.key },
      kind: 'all',
      metadata: {
        generation_mode: 'minimax_h3_ref2v',
        minimax_h3_references: [{ detail: 'max', image_name: 'ref.png', kind: 'image' }],
        minimax_h3_transformer_model: { key: checkpoint.key },
        model: { key: 'h3-install-gone' },
        num_frames: 124,
      },
      models: [panelCheckpoint, checkpoint],
    });

    expect(result?.values.model?.key).toBe(checkpoint.key);
    expect(result?.fields).toContain('model');
    expect(result?.values.h3TransformerModel).toBeNull();
    expect(result?.mediaNames.references).toHaveLength(1);
  });

  it('drops a corrupt transformer-override recording that names a non-main', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: {
        generation_mode: 'minimax_h3_ref2v',
        minimax_h3_references: [{ detail: 'max', image_name: 'ref.png', kind: 'image' }],
        minimax_h3_transformer_model: { key: ref2vTurbo.key },
        model: { key: install.key },
        num_frames: 124,
      },
      models: catalog,
    });

    expect(result?.values.model?.key).toBe(install.key);
    expect(result?.values.h3TransformerModel).toBeNull();
  });

  it('reports the source video alongside the references for a reference-extend recording', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: {
        generation_mode: 'minimax_h3_ref2v',
        minimax_h3_component_source: { key: install.key },
        minimax_h3_references: [
          { conditioning: 'video_audio', end_frame: 400, kind: 'video', start_frame: 260, video_name: 'long.mp4' },
        ],
        model: { key: checkpoint.key },
        num_frames: 124,
        source_video: { video_name: 'long.mp4' },
        source_video_end_frame: 400,
        source_video_start_frame: 10,
      },
      models: catalog,
    });

    expect(result?.fields).toContain('media');
    expect(result?.mediaNames.references).toHaveLength(1);
    expect(result?.mediaNames.sourceVideoName).toBe('long.mp4');
    expect(result?.mediaNames.sourceVideoTrim).toEqual({ endFrame: 400, startFrame: 10 });
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

  it('resolves a model recorded on another install by hash when its key is unknown here', () => {
    const hashed = { ...WAN_T2V, hash: 'blake3:abc' };
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: wanMetadata({
        model: { base: 'wan', hash: 'blake3:abc', key: 'foreign-key', name: 'Other name', type: 'main' },
      }),
      models: [hashed, h3Model()],
    });

    expect(result?.fields).toContain('model');
    expect(result?.values.model?.key).toBe(WAN_T2V.key);
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

  it('does not advertise a recall for a model reference nothing can resolve', () => {
    expect(getVideoRecallCapabilities({ generation_mode: 'wan_t2v', model: { name: 'only a name' } })).toEqual(
      EMPTY_VIDEO_RECALL_CAPABILITIES
    );
    expect(getVideoRecallCapabilities({ generation_mode: 'wan_t2v', model: { hash: 'blake3:x' } }).remix).toBe(true);
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

  it('recalls the context length a continuation was made with', () => {
    // Not recoverable from anything else in the record: the output length folds the source, the
    // generated half and the crossfade together, so without this a recall silently reinstates the
    // default 17 and reproduces a different run.
    const result = buildVideoRecallSettings({
      currentValues: createDefaultVideoWidgetValues([LTX2_DEV]),
      kind: 'all',
      metadata: ltx2Metadata({ ltx2_context_frames: 49 }),
      models: catalog,
    });

    expect(result?.values).toMatchObject({ ltx2ExtendContextFrames: 49 });
    expect(result?.fields).toEqual(expect.arrayContaining(['extendContext']));
  });

  it('snaps a recalled context off the grid and ignores one on a non-LTX-2 clip', () => {
    const offGrid = buildVideoRecallSettings({
      currentValues: createDefaultVideoWidgetValues([LTX2_DEV]),
      kind: 'all',
      metadata: ltx2Metadata({ ltx2_context_frames: 30 }),
      models: catalog,
    });
    // A Wan clip carrying the key (hand-edited metadata) must not write an LTX-2-only setting.
    const wrongFamily = buildVideoRecallSettings({
      currentValues: createDefaultVideoWidgetValues([WAN_T2V]),
      kind: 'all',
      metadata: { ...wanMetadata(), ltx2_context_frames: 49 },
      models: [WAN_T2V, LIGHTNING_HIGH, LIGHTNING_LOW],
    });

    expect(offGrid?.values).toMatchObject({ ltx2ExtendContextFrames: 25 });
    expect(wrongFamily?.fields).not.toContain('extendContext');
  });

  it('recalls an ordinary clip into a panel that currently has the accelerator on', () => {
    // The accelerator hides Steps and every guidance scale, and the policy that decides what recall
    // may write is derived from the panel's CURRENT state. Asked as-is it reports those controls
    // invisible and drops all of them, leaving the accelerator's 8 / 1 / 1 / 1 / 0 on screen as the
    // recalled clip's values -- numbers that clip never used, with nothing saying they were dropped.
    const accelerated = {
      ...createDefaultVideoWidgetValues([LTX2_DEV]),
      acceleratorEnabled: true,
      acceleratorLoraKeys: ['ltx2-distilled'],
      audioCfgScale: 1,
      cfgScale: 1,
      modalityScale: 1,
      steps: 8,
      stgScale: 0,
    };
    const result = buildVideoRecallSettings({
      currentValues: accelerated,
      kind: 'all',
      metadata: ltx2Metadata(),
      models: catalog,
    });

    expect(result?.values).toMatchObject({
      audioCfgScale: 7,
      cfgScale: 3,
      modalityScale: 3,
      steps: 30,
      stgScale: 1,
    });
    expect(result?.fields).toEqual(expect.arrayContaining(['steps', 'cfg']));
  });

  it('writes no control a partial recall leaves hidden behind the panel accelerator', () => {
    const accelerated = {
      ...createDefaultVideoWidgetValues([LTX2_DEV]),
      acceleratorEnabled: true,
      acceleratorLoraKeys: ['ltx2-distilled'],
      cfgScale: 1,
      steps: 8,
    };
    const result = buildVideoRecallSettings({
      currentValues: accelerated,
      kind: 'all',
      metadata: { positive_prompt: 'p', steps: 30, cfg_scale: 4 },
      models: catalog,
      partial: true,
      requireGenerationMode: false,
    });

    expect(result?.fields).toEqual(['prompts']);
    expect(result?.values).toMatchObject({ acceleratorEnabled: true, cfgScale: 1, steps: 8 });
  });

  it('treats both LTX-2 modes as recallable video metadata', () => {
    // The mode id is the gate for every Recall button; a string the set does not know silently
    // hides them all, with nothing failing.
    for (const mode of ['ltx2_t2v', 'ltx2_i2v']) {
      expect(isVideoGenerationMetadata({ generation_mode: mode, seed: 1 })).toBe(true);
      expect(getVideoRecallCapabilities(ltx2Metadata({ generation_mode: mode }))).toEqual({
        all: true,
        prompts: true,
        remix: true,
        seed: true,
      });
    }
  });

  it('recalls a two-stage run as the preset that produces its canvas', () => {
    // A two-stage run records its final canvas, and no single-stage preset resolves to it -- so the
    // preset comes back from the size alone, without the metadata having to name the stage count.
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

  it('reproduces a dev clip that ran without a negative prompt, from a distilled panel', () => {
    // Switching to dev seeds the release's list into a panel that carries none -- which is right
    // when the user picks the model, and wrong here: the clip recorded an empty negative prompt and
    // a recall has to re-run what was generated, not what the panel would default to.
    const onDistilled = createDefaultVideoWidgetValues([LTX2_DISTILLED]);

    const result = buildVideoRecallSettings({
      currentValues: onDistilled,
      kind: 'all',
      metadata: ltx2Metadata({ negative_prompt: '' }),
      models: catalog,
    });

    expect(result?.values.model).toMatchObject({ key: LTX2_DEV.key });
    expect(result?.values.cfgScale).toBe(3);
    expect(result?.values.negativePrompt).toBe('');
  });

  it('restores the model, both component slots and every guidance scale', () => {
    const result = buildVideoRecallSettings({ currentValues, kind: 'all', metadata: ltx2Metadata(), models: catalog });

    expect(result?.values.model).toMatchObject({ key: LTX2_DEV.key });
    expect(result?.values.componentSourceModel).toMatchObject({ key: LTX2_COMPONENTS.key });
    expect(result?.values.ltx2TextEncoderModel).toMatchObject({ key: LTX2_ENCODER.key });
    expect(result?.values).toMatchObject({
      audioCfgScale: 7,
      cfgScale: 3,
      fps: 24,
      modalityScale: 3,
      numFrames: 121,
      seed: 99,
      steps: 30,
      stgScale: 1,
    });
    expect(result?.fields).toEqual(expect.arrayContaining(['model', 'components', 'cfg', 'frames', 'steps']));
  });

  it('recalls the size back onto the preset that produced it', () => {
    const result = buildVideoRecallSettings({ currentValues, kind: 'all', metadata: ltx2Metadata(), models: catalog });

    expect(result?.values.targetResolution).toBe('704p');
    expect(result?.values.aspectRatioId).toBe('16:9');
  });

  it('does not put a step count on a checkpoint whose schedule is fixed', () => {
    // The distilled schedule ignores whatever reaches it, so a recalled 30 would leave a disabled
    // control showing a number the run will not use — and re-record it on the next generation.
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: ltx2Metadata({ model: LTX2_DISTILLED, steps: 30 }),
      models: catalog,
    });

    expect(result?.values.model).toMatchObject({ key: LTX2_DISTILLED.key });
    expect(result?.values.steps).toBe(8);
    expect(result?.fields).not.toContain('steps');
  });

  it('reproduces a run made with the negative prompt switched off', () => {
    // Whether the negative prompt is on changes the LTX-2 graph — it decides whether an
    // unconditional pass runs at all — and nothing records that bit directly. What is recorded is
    // the guidance the run actually used, which is the same thing: both scales at 1 mean no
    // unconditional pass, so recalling them rebuilds the same graph even with the prompt back on.
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: ltx2Metadata({ cfg_scale: 1, ltx2_audio_cfg_scale: 1 }),
      models: catalog,
    });

    expect(result?.values.cfgScale).toBe(1);
    expect(result?.values.audioCfgScale).toBe(1);
    expect(getVideoPromptPolicy(result!.values.model!, result!.values).negativeUsedInGraph).toBe(false);
  });

  it('drops the per-modality scales when recalled onto a family without them', () => {
    const result = buildVideoRecallSettings({
      currentValues,
      kind: 'all',
      metadata: { ...ltx2Metadata(), model: WAN_T2V, generation_mode: 'wan_t2v' },
      models: catalog,
    });

    expect(result?.values.audioCfgScale).toBeNull();
    expect(result?.values.stgScale).toBeNull();
    expect(result?.values.modalityScale).toBeNull();
  });
});
