import type { GenerationModelCatalogItem, MainModelConfig } from '@features/generation/contracts';

import { describe, expect, it } from 'vitest';

import type { VideoSettings } from './types';

import {
  findMiniMaxH3TurboLora,
  findWanLightningLoraPair,
  getDefaultVideoSettings,
  getAcceleratorToggleResult,
  getVideoComponentSectionPolicy,
  getVideoDimensions,
  getVideoModelAvailabilityReasons,
  getVideoModelPolicy,
  getVideoModelSelectionResult,
  getVideoModes,
  getVideoPromptPolicy,
  getVideoValidationReasons,
  getWanExpertWiringWarning,
  isSupportedVideoModel,
  isValidVideoNumFrames,
  snapVideoNumFrames,
  WAN_LIGHTNING_ACCELERATOR,
} from './videoPolicies';

const wanModel = (variant: string, format = 'gguf_quantized', key = `wan-${variant}-${format}`): MainModelConfig => ({
  base: 'wan',
  format,
  key,
  name: `Wan 2.2 ${variant}`,
  type: 'main',
  variant,
});

const h3Model = (format = 'diffusers', key = `h3-${format}`): MainModelConfig => ({
  base: 'minimax-h3',
  format,
  key,
  name: 'MiniMax H3',
  type: 'main',
  variant: 'fl2va',
});

const lora = (name: string, variant: string | null = 'a14b', key = `lora-${name}`): GenerationModelCatalogItem => ({
  base: 'wan',
  key,
  name,
  type: 'lora',
  variant,
});

const LIGHTNING_T2V_HIGH = lora('Wan 2.2 T2V Lightning High Noise (4-step)');
const LIGHTNING_T2V_LOW = lora('Wan 2.2 T2V Lightning Low Noise (4-step)');
const LIGHTNING_I2V_HIGH = lora('Wan 2.2 I2V Lightning High Noise (4-step)');
const LIGHTNING_I2V_LOW = lora('Wan 2.2 I2V Lightning Low Noise (4-step)');

const FIRST_FRAME = { height: 1080, image_name: 'first.png', width: 1920 };
const LAST_FRAME = { height: 1080, image_name: 'last.png', width: 1920 };
const SOURCE_VIDEO = {
  endFrame: 79,
  fps: 16,
  height: 480,
  numFrames: 81,
  startFrame: 0,
  video_name: 'clip.mp4',
  width: 832,
};

const settingsFor = (model?: MainModelConfig, overrides: Partial<VideoSettings> = {}): VideoSettings => ({
  ...getDefaultVideoSettings(model),
  ...overrides,
});

describe('isSupportedVideoModel', () => {
  it('accepts Wan mains in any format', () => {
    expect(isSupportedVideoModel(wanModel('t2v_a14b', 'gguf_quantized'))).toBe(true);
    expect(isSupportedVideoModel(wanModel('i2v_a14b', 'diffusers'))).toBe(true);
    expect(isSupportedVideoModel(wanModel('ti2v_5b', 'checkpoint'))).toBe(true);
  });

  it('accepts MiniMax H3 mains only in Diffusers format', () => {
    expect(isSupportedVideoModel(h3Model('diffusers'))).toBe(true);
    // Single-file H3 checkpoints are transformer overrides, not selectable mains.
    expect(isSupportedVideoModel(h3Model('checkpoint'))).toBe(false);
  });

  it('rejects non-video bases and non-main types', () => {
    expect(isSupportedVideoModel({ base: 'sdxl', format: 'checkpoint', type: 'main' })).toBe(false);
    expect(isSupportedVideoModel({ base: 'wan', format: 'checkpoint', type: 'lora' })).toBe(false);
  });
});

describe('capabilities matrix', () => {
  it('gates conditioning modes per Wan variant', () => {
    expect(getVideoModes(wanModel('t2v_a14b'))).toEqual(['txt2vid']);
    expect(getVideoModes(wanModel('i2v_a14b'))).toEqual(['first-frame', 'first-last', 'extend']);
    expect(getVideoModes(wanModel('ti2v_5b'))).toEqual(['txt2vid', 'first-frame', 'extend']);
  });

  it('gives MiniMax H3 every mode including last-frame-only', () => {
    expect(getVideoModes(h3Model())).toEqual(['txt2vid', 'first-frame', 'last-frame', 'first-last', 'extend']);
  });

  it('grants unknown Wan variants permissive A14B capabilities', () => {
    const unknown = wanModel('some_future_variant');

    expect(getVideoModes(unknown)).toContain('txt2vid');
    expect(getVideoModes(unknown)).toContain('extend');
  });

  it('shapes the sampling UI per family', () => {
    const a14b = getVideoModelPolicy(wanModel('i2v_a14b'), settingsFor(wanModel('i2v_a14b')));
    const ti2v = getVideoModelPolicy(wanModel('ti2v_5b'), settingsFor(wanModel('ti2v_5b')));
    const h3 = getVideoModelPolicy(h3Model(), settingsFor(h3Model()));

    expect(a14b.ui).toMatchObject({ cfgLowNoiseVisible: true, cfgVisible: true, fpsVisible: true });
    expect(a14b.ui.accelerator).toMatchObject({ label: 'Lightning', steps: 4 });
    expect(a14b.frames).toMatchObject({ defaultValue: 81, kind: 'grid', step: 4 });

    expect(ti2v.ui).toMatchObject({ accelerator: null, cfgLowNoiseVisible: false, cfgVisible: true });

    expect(h3.ui).toMatchObject({
      audioOutput: true,
      cfgLowNoiseVisible: false,
      cfgVisible: false,
      fpsVisible: false,
    });
    expect(h3.ui.accelerator).toMatchObject({ label: 'Turbo', steps: 6 });
    expect(h3.frames.kind).toBe('choices');
    expect(h3.fps).toMatchObject({ defaultValue: 24, editable: false });
    expect(h3.defaults.steps).toBe(50);
  });
});

describe('frame snapping per model', () => {
  it('snaps to the Wan 4n + 1 grid and the H3 choice list respectively', () => {
    expect(snapVideoNumFrames(wanModel('t2v_a14b'), 80)).toBe(81);
    expect(snapVideoNumFrames(h3Model(), 81)).toBe(90);
    expect(isValidVideoNumFrames(wanModel('t2v_a14b'), 81)).toBe(true);
    expect(isValidVideoNumFrames(h3Model(), 81)).toBe(false);
    expect(isValidVideoNumFrames(h3Model(), 124)).toBe(true);
  });
});

describe('getVideoDimensions', () => {
  it('derives from the aspect-ratio preset in text-to-video', () => {
    const model = wanModel('t2v_a14b');

    expect(getVideoDimensions(model, settingsFor(model, { aspectRatioId: '16:9', targetResolution: '480p' }))).toEqual({
      height: 480,
      source: 'aspect-ratio',
      width: 848,
    });
  });

  it('derives from conditioning media once it is set, ignoring the preset ratio', () => {
    const model = wanModel('i2v_a14b');
    const settings = settingsFor(model, {
      aspectRatioId: '1:1',
      firstFrameImage: FIRST_FRAME,
      targetResolution: '720p',
    });

    expect(getVideoDimensions(model, settings)).toEqual({ height: 720, source: 'first-frame', width: 1280 });
  });

  it('prefers the source video over frame images, and last frame over the preset', () => {
    const model = h3Model();

    expect(
      getVideoDimensions(model, settingsFor(model, { lastFrameImage: LAST_FRAME, sourceVideo: SOURCE_VIDEO }))?.source
    ).toBe('source-video');
    expect(getVideoDimensions(model, settingsFor(model, { lastFrameImage: LAST_FRAME }))?.source).toBe('last-frame');
  });

  it('applies the H3 canvas policy including the area cap', () => {
    const model = h3Model();

    expect(getVideoDimensions(model, settingsFor(model, { aspectRatioId: '16:9' }))).toEqual({
      height: 768,
      source: 'aspect-ratio',
      width: 1344,
    });
  });

  it('coerces a stale target resolution from another family before resolving', () => {
    const model = h3Model();
    const settings = settingsFor(model, { targetResolution: '480p' });

    expect(getVideoDimensions(model, settings)).toEqual({ height: 768, source: 'aspect-ratio', width: 1344 });
  });

  it('returns null for media outside the H3 aspect range', () => {
    const model = h3Model();
    const settings = settingsFor(model, {
      firstFrameImage: { height: 100, image_name: 'strip.png', width: 500 },
    });

    expect(getVideoDimensions(model, settings)).toBeNull();
  });
});

describe('getVideoPromptPolicy', () => {
  const promptSettings = (overrides: Partial<Parameters<typeof getVideoPromptPolicy>[1]> = {}) => ({
    cfgScale: 5,
    cfgScaleLowNoise: null,
    negativePromptEnabled: true,
    wanLowNoiseModel: null,
    ...overrides,
  });

  it('is cfg-gated for Wan', () => {
    const model = wanModel('t2v_a14b');

    const atCfg5 = getVideoPromptPolicy(model, promptSettings());
    const atCfg1 = getVideoPromptPolicy(model, promptSettings({ cfgScale: 1 }));

    expect(atCfg5).toMatchObject({ negativeUsedInGraph: true, negativeVisible: true });
    expect(atCfg1).toMatchObject({ negativeUsedInGraph: false, negativeVisible: true });
    expect(atCfg1.negativeHelpText).toMatch(/CFG/);
  });

  it('counts low-noise CFG > 1 as CFG in use, matching wan_video_denoise do_cfg', () => {
    const singleFileMain = wanModel('i2v_a14b', 'gguf_quantized');
    const lowExpert = wanModel('i2v_a14b', 'checkpoint', 'low-expert');
    const base = promptSettings({ cfgScale: 1, cfgScaleLowNoise: 4 });

    // Low expert wired → the low-noise half runs CFG and consumes the negative prompt.
    expect(getVideoPromptPolicy(singleFileMain, { ...base, wanLowNoiseModel: lowExpert }).negativeUsedInGraph).toBe(
      true
    );
    // Diffusers A14B mains bundle transformer_2, so the low half exists without a wired expert.
    expect(getVideoPromptPolicy(wanModel('i2v_a14b', 'diffusers'), base).negativeUsedInGraph).toBe(true);
    // No second expert at all → the low CFG value never runs.
    expect(getVideoPromptPolicy(singleFileMain, base).negativeUsedInGraph).toBe(false);
    // TI2V-5B has no low-noise half regardless of the stale value.
    expect(getVideoPromptPolicy(wanModel('ti2v_5b', 'diffusers'), base).negativeUsedInGraph).toBe(false);
  });

  it('never shows a negative prompt for MiniMax H3', () => {
    expect(getVideoPromptPolicy(h3Model(), promptSettings())).toMatchObject({
      negativeUsedInGraph: false,
      negativeVisible: false,
    });
  });
});

describe('Lightning', () => {
  const model = wanModel('t2v_a14b');
  const catalog = [LIGHTNING_T2V_HIGH, LIGHTNING_T2V_LOW, LIGHTNING_I2V_HIGH, LIGHTNING_I2V_LOW, lora('Style LoRA')];

  it('finds the installed pair, preferring the main model’s family', () => {
    expect(findWanLightningLoraPair(catalog, 't2v_a14b')).toMatchObject({
      high: { key: LIGHTNING_T2V_HIGH.key },
      low: { key: LIGHTNING_T2V_LOW.key },
    });
    expect(findWanLightningLoraPair(catalog, 'i2v_a14b')).toMatchObject({
      high: { key: LIGHTNING_I2V_HIGH.key },
      low: { key: LIGHTNING_I2V_LOW.key },
    });
  });

  it('falls back across families and rejects incomplete pairs', () => {
    expect(findWanLightningLoraPair([LIGHTNING_T2V_HIGH, LIGHTNING_T2V_LOW], 'i2v_a14b')).not.toBeNull();
    expect(findWanLightningLoraPair([LIGHTNING_T2V_HIGH], 't2v_a14b')).toBeNull();
    // 5B mains never match the A14B Lightning LoRAs.
    expect(findWanLightningLoraPair(catalog, 'ti2v_5b')).toBeNull();
  });

  it('matches high/low only as delimited tokens, not inside ordinary words', () => {
    // "Slow" must not read as a low-noise expert; the pair is incomplete.
    expect(findWanLightningLoraPair([LIGHTNING_T2V_HIGH, lora('Wan Slow Motion Lightning')], 't2v_a14b')).toBeNull();
    expect(findWanLightningLoraPair([lora('Thigh-Focus Lightning'), LIGHTNING_T2V_LOW], 't2v_a14b')).toBeNull();
    // Underscore-delimited release filenames still match.
    expect(
      findWanLightningLoraPair(
        [lora('Wan2.2-Lightning_high_noise_model'), lora('Wan2.2-Lightning_low_noise_model')],
        't2v_a14b'
      )
    ).not.toBeNull();
  });

  it("does not let 'i2v' score inside 'TI2V' when preferring a family", () => {
    // A mistagged (variant-less) TI2V-named pair listed first must not outrank
    // the true I2V pair for an I2V main.
    const ti2vHigh = lora('Wan Lightning TI2V-5B High Noise', null);
    const ti2vLow = lora('Wan Lightning TI2V-5B Low Noise', null);

    expect(
      findWanLightningLoraPair([ti2vHigh, ti2vLow, LIGHTNING_I2V_HIGH, LIGHTNING_I2V_LOW], 'i2v_a14b')
    ).toMatchObject({
      high: { key: LIGHTNING_I2V_HIGH.key },
      low: { key: LIGHTNING_I2V_LOW.key },
    });
  });

  it('toggling on patches sampling and adds the pair to the visible LoRA list', () => {
    const base = settingsFor(model, {
      cfgScale: 5,
      cfgScaleLowNoise: 4,
      acceleratorEnabled: false,
      loras: [],
      steps: 40,
    });
    const result = getAcceleratorToggleResult(base, model, catalog, true);

    expect(result.missingLoras).toBe(false);
    expect(result.settings).toMatchObject({ acceleratorEnabled: true, cfgScale: 1, cfgScaleLowNoise: 1, steps: 4 });
    expect(result.settings.loras.map((entry) => entry.model.key)).toEqual([
      LIGHTNING_T2V_HIGH.key,
      LIGHTNING_T2V_LOW.key,
    ]);
  });

  it('toggling off removes only the Lightning LoRAs and restores model defaults', () => {
    const on = getAcceleratorToggleResult(settingsFor(model), model, catalog, true).settings;
    const withStyle = { ...on, loras: [...on.loras, { isEnabled: true, model: lora('Style LoRA'), weight: 0.5 }] };
    const off = getAcceleratorToggleResult(withStyle as VideoSettings, model, catalog, false).settings;

    expect(off.acceleratorEnabled).toBe(false);
    expect(off).toMatchObject({ cfgScale: 5, cfgScaleLowNoise: 4, steps: 40 });
    expect(off.loras.map((entry) => entry.model.name)).toEqual(['Style LoRA']);
  });

  it('reports a missing pair instead of enabling silently', () => {
    const result = getAcceleratorToggleResult(settingsFor(model), model, [], true);

    expect(result.missingLoras).toBe(true);
    expect(result.settings.acceleratorEnabled).toBe(false);
  });

  it('clears a stale enabled flag when the pair has vanished from the catalog', () => {
    const stale = settingsFor(model, { acceleratorEnabled: true, steps: 4 });
    const result = getAcceleratorToggleResult(stale, model, [], true);

    expect(result.missingLoras).toBe(true);
    expect(result.settings.acceleratorEnabled).toBe(false);
  });

  it('defaults Lightning on for A14B when the pair is installed, off otherwise', () => {
    const withPair = getDefaultVideoSettings(model, catalog);
    const withoutPair = getDefaultVideoSettings(model, []);
    const h3Defaults = getDefaultVideoSettings(h3Model(), catalog);

    expect(withPair.acceleratorEnabled).toBe(true);
    expect(withPair.steps).toBe(WAN_LIGHTNING_ACCELERATOR.steps);
    expect(withoutPair).toMatchObject({ cfgScale: 5, acceleratorEnabled: false, steps: 40 });
    // The catalog holds no H3 Turbo LoRA, so H3 falls back to its slow defaults.
    expect(h3Defaults).toMatchObject({ fps: 24, acceleratorEnabled: false, numFrames: 124, steps: 50 });
  });
});

describe('MiniMax H3 Turbo', () => {
  const TURBO = { base: 'minimax-h3', key: 'turbo', name: 'MiniMax H3 Turbo LoRA', type: 'lora' as const };

  it('finds the installed Turbo LoRA by name, ignoring Wan Lightning models', () => {
    expect(findMiniMaxH3TurboLora([LIGHTNING_T2V_HIGH, TURBO])).toMatchObject({ key: 'turbo' });
    expect(findMiniMaxH3TurboLora([LIGHTNING_T2V_HIGH, LIGHTNING_T2V_LOW])).toBeNull();
  });

  it('defaults Turbo on for H3 when installed, matching the bundled templates (steps 6)', () => {
    const defaults = getDefaultVideoSettings(h3Model(), [TURBO]);

    expect(defaults).toMatchObject({ acceleratorEnabled: true, steps: 6 });
    expect(defaults.loras.map((entry) => entry.model.key)).toEqual(['turbo']);
  });

  it('toggling off restores the slow 50-step default and removes the Turbo LoRA', () => {
    const on = getDefaultVideoSettings(h3Model(), [TURBO]);
    const off = getAcceleratorToggleResult(on, h3Model(), [TURBO], false).settings;

    expect(off).toMatchObject({ acceleratorEnabled: false, steps: 50 });
    expect(off.loras).toEqual([]);
  });

  it('prefers the family-named Turbo repack over a look-alike, deterministically', () => {
    const turboRider = { base: 'minimax-h3', key: 'rider', name: 'Turbo Rider', type: 'lora' as const };

    expect(findMiniMaxH3TurboLora([turboRider, TURBO])).toMatchObject({ key: 'turbo' });
    expect(findMiniMaxH3TurboLora([TURBO, turboRider])).toMatchObject({ key: 'turbo' });
  });

  it('never strips a user LoRA that merely shares an accelerator-style name', () => {
    const turboRider = { base: 'minimax-h3', key: 'rider', name: 'Turbo Rider', type: 'lora' as const };
    const model = h3Model();
    const catalog = [TURBO, turboRider];
    const on = getAcceleratorToggleResult(settingsFor(model), model, catalog, true).settings;
    const withRider = {
      ...on,
      loras: [...on.loras, { isEnabled: true, model: turboRider as never, weight: 0.8 }],
    };
    const off = getAcceleratorToggleResult(withRider as VideoSettings, model, catalog, false).settings;

    // Only the recorded Turbo entry is removed; "Turbo Rider" is the user's.
    expect(off.loras.map((entry) => entry.model.key)).toEqual(['rider']);
    expect(off.acceleratorLoraKeys).toEqual([]);
  });

  it('carries the fast-path intent across a family switch (Lightning → Turbo)', () => {
    const wan = wanModel('t2v_a14b');
    const catalog = [LIGHTNING_T2V_HIGH, LIGHTNING_T2V_LOW, TURBO];
    const lightningOn = getAcceleratorToggleResult(settingsFor(wan), wan, catalog, true).settings;
    const result = getVideoModelSelectionResult({ currentSettings: lightningOn, model: h3Model(), models: catalog });

    expect(result.settings.acceleratorEnabled).toBe(true);
    expect(result.settings.steps).toBe(6);
    expect(result.settings.loras.map((entry) => entry.model.key)).toEqual(['turbo']);
    expect(result.clearedLabels).toContain('Acceleration');
  });
});

describe('component section policy', () => {
  it('requires VAE and Wan T5 for split Wan models without a component source', () => {
    const model = wanModel('i2v_a14b', 'gguf_quantized');
    const settings = settingsFor(model);
    const policy = getVideoComponentSectionPolicy(model, settings);

    expect(policy.defaultOpen).toBe(true);
    expect(policy.slots.map((slot) => slot.key)).toEqual([
      'componentSourceModel',
      'vae',
      'wanT5EncoderModel',
      'wanLowNoiseModel',
    ]);

    const reasons = getVideoValidationReasons(model, { ...settings, firstFrameImage: FIRST_FRAME });

    expect(reasons).toContain('Video needs a VAE for Wan models.');
    expect(reasons).toContain('Video needs a Wan T5 Encoder for Wan models.');
  });

  it('is satisfied by a Diffusers component source or a Diffusers main', () => {
    const gguf = wanModel('i2v_a14b', 'gguf_quantized');
    const withSource = settingsFor(gguf, {
      componentSourceModel: wanModel('i2v_a14b', 'diffusers'),
      firstFrameImage: FIRST_FRAME,
    });

    expect(getVideoValidationReasons(gguf, withSource)).toEqual([]);

    const diffusers = wanModel('i2v_a14b', 'diffusers');

    expect(getVideoValidationReasons(diffusers, settingsFor(diffusers, { firstFrameImage: FIRST_FRAME }))).toEqual([]);
  });

  it('hides the low-noise expert slot for TI2V-5B and for Diffusers mains', () => {
    const ti2v = wanModel('ti2v_5b', 'checkpoint');

    expect(getVideoComponentSectionPolicy(ti2v, settingsFor(ti2v)).slots.map((slot) => slot.key)).not.toContain(
      'wanLowNoiseModel'
    );

    // A Diffusers A14B main bundles transformer_2; the loader ignores the input.
    const diffusers = wanModel('i2v_a14b', 'diffusers');

    expect(
      getVideoComponentSectionPolicy(diffusers, settingsFor(diffusers)).slots.map((slot) => slot.key)
    ).not.toContain('wanLowNoiseModel');
  });

  it('requires both single-file overrides for a components-only MiniMax H3 install', () => {
    // The slim "MiniMax H3 Components" folder carries tokenizer/processor/VAEs
    // but no transformer or text-encoder weights (backend `components_only`):
    // without the gate, Invoke enables and the loader fails mid-generation.
    const componentsOnly = { ...h3Model(), components_only: true };
    const bare = settingsFor(componentsOnly);

    expect(getVideoComponentSectionPolicy(componentsOnly, bare).defaultOpen).toBe(true);

    const reasons = getVideoValidationReasons(componentsOnly, bare);

    expect(reasons).toContainEqual(expect.stringContaining('select a single-file Transformer'));
    expect(reasons).toContainEqual(expect.stringContaining('select a single-file Text encoder'));

    // Both overrides selected → the gate opens.
    const filled = settingsFor(componentsOnly, {
      h3TextEncoderModel: { base: 'minimax-h3', key: 'h3-te-int8', name: 'H3 Text Encoder', type: 'qwen3_vl_encoder' },
      h3TransformerModel: {
        base: 'minimax-h3',
        format: 'checkpoint',
        key: 'h3-tf-int8',
        name: 'H3 Transformer',
        type: 'main',
      },
    });

    expect(getVideoValidationReasons(componentsOnly, filled)).toEqual([]);

    // One override alone is not enough.
    expect(getVideoValidationReasons(componentsOnly, { ...filled, h3TextEncoderModel: null })).toContainEqual(
      expect.stringContaining('select a single-file Text encoder')
    );

    // A full Diffusers install keeps both slots optional.
    const full = h3Model();

    expect(getVideoValidationReasons(full, settingsFor(full))).toEqual([]);
    expect(getVideoComponentSectionPolicy(full, settingsFor(full)).defaultOpen).toBe(false);
  });

  it('offers the component-source slot only for single-file mains, never listing the main itself', () => {
    const diffusers = wanModel('i2v_a14b', 'diffusers');

    expect(
      getVideoComponentSectionPolicy(diffusers, settingsFor(diffusers)).slots.map((slot) => slot.key)
    ).not.toContain('componentSourceModel');

    const gguf = wanModel('i2v_a14b', 'gguf_quantized');
    const settings = settingsFor(gguf);
    const slot = getVideoComponentSectionPolicy(gguf, settings).slots.find((s) => s.key === 'componentSourceModel');
    const ctx = { model: gguf, selectedComponents: settings, settings };

    expect(slot?.filter?.(wanModel('i2v_a14b', 'diffusers'), ctx)).toBe(true);
    // A (hypothetical) diffusers-format selected main must not list itself.
    expect(slot?.filter?.({ ...gguf, format: 'diffusers' }, { ...ctx, model: { ...gguf, format: 'diffusers' } })).toBe(
      false
    );
  });

  it('constrains the low-noise expert to a different single-file model of the same variant', () => {
    const model = wanModel('i2v_a14b', 'gguf_quantized');
    const settings = settingsFor(model);
    const slot = getVideoComponentSectionPolicy(model, settings).slots.find((s) => s.key === 'wanLowNoiseModel');
    const ctx = { model, selectedComponents: settings, settings };

    expect(slot?.filter?.(wanModel('i2v_a14b', 'checkpoint', 'low'), ctx)).toBe(true);
    // Same model as the main is rejected (loader raises on identical keys).
    expect(slot?.filter?.(model, ctx)).toBe(false);
    // Diffusers-format experts are rejected (loader requires single-file).
    expect(slot?.filter?.(wanModel('i2v_a14b', 'diffusers'), ctx)).toBe(false);
    // Cross-variant experts are rejected (loader requires exact variant equality).
    expect(slot?.filter?.(wanModel('t2v_a14b', 'checkpoint'), ctx)).toBe(false);
    // Unknown variants stay allowed — the backend probe is the authority.
    expect(slot?.filter?.({ ...wanModel('i2v_a14b', 'checkpoint', 'untagged'), variant: null }, ctx)).toBe(true);
  });

  it('matches the standalone VAE to the main variant by latent channels', () => {
    const a14b = wanModel('i2v_a14b', 'gguf_quantized');
    const ti2v = wanModel('ti2v_5b', 'checkpoint');
    const vae16 = { base: 'wan', key: 'vae16', latent_channels: 16, name: 'Wan 2.1 VAE', type: 'vae' };
    const vae48 = { base: 'wan', key: 'vae48', latent_channels: 48, name: 'Wan 2.2 VAE', type: 'vae' };
    const vaeUnknown = { base: 'wan', key: 'vae?', name: 'Mystery Wan VAE', type: 'vae' };

    const slotFor = (model: MainModelConfig) => {
      const settings = settingsFor(model);

      return {
        ctx: { model, selectedComponents: settings, settings },
        slot: getVideoComponentSectionPolicy(model, settings).slots.find((s) => s.key === 'vae'),
      };
    };

    const forA14b = slotFor(a14b);
    const forTi2v = slotFor(ti2v);

    expect(forA14b.slot?.filter?.(vae16, forA14b.ctx)).toBe(true);
    expect(forA14b.slot?.filter?.(vae48, forA14b.ctx)).toBe(false);
    expect(forTi2v.slot?.filter?.(vae48, forTi2v.ctx)).toBe(true);
    expect(forTi2v.slot?.filter?.(vae16, forTi2v.ctx)).toBe(false);
    // Configs without the field (open union) stay allowed on both.
    expect(forA14b.slot?.filter?.(vaeUnknown, forA14b.ctx)).toBe(true);
    expect(forTi2v.slot?.filter?.(vaeUnknown, forTi2v.ctx)).toBe(true);
  });

  it('a cross-family component source covers the encoder but not the VAE', () => {
    // TI2V-5B main with an A14B Diffusers source: UMT5 is shared (encoder
    // satisfied) but the VAE is family-bound, so a VAE is still required.
    const model = wanModel('ti2v_5b', 'checkpoint');
    const settings = settingsFor(model, { componentSourceModel: wanModel('i2v_a14b', 'diffusers') });
    const reasons = getVideoValidationReasons(model, settings);

    expect(reasons).toContain('Video needs a VAE for Wan models.');
    expect(reasons).not.toContain('Video needs a Wan T5 Encoder for Wan models.');
  });

  it('offers only optional single-file overrides for MiniMax H3', () => {
    const model = h3Model();
    const policy = getVideoComponentSectionPolicy(model, settingsFor(model));

    expect(policy.defaultOpen).toBe(false);
    expect(policy.slots.map((slot) => slot.key)).toEqual(['h3TransformerModel', 'h3TextEncoderModel']);
    expect(policy.slots.every((slot) => !slot.required)).toBe(true);

    const transformerSlot = policy.slots[0];
    const ctx = { model, selectedComponents: settingsFor(model), settings: settingsFor(model) };

    expect(transformerSlot?.filter?.(h3Model('checkpoint'), ctx)).toBe(true);
    expect(transformerSlot?.filter?.(h3Model('diffusers'), ctx)).toBe(false);
  });
});

describe('getWanExpertWiringWarning', () => {
  const tagged = (variant: string, expert: 'high' | 'low' | 'none', key: string): MainModelConfig => ({
    ...wanModel(variant, 'gguf_quantized', key),
    expert,
  });

  it('flags swapped, mislabeled, and single-low wirings on single-file A14B mains', () => {
    expect(getWanExpertWiringWarning(tagged('i2v_a14b', 'low', 'm'), tagged('i2v_a14b', 'high', 'l'))).toEqual({
      kind: 'swapped',
    });
    expect(getWanExpertWiringWarning(tagged('i2v_a14b', 'none', 'm'), tagged('i2v_a14b', 'high', 'l'))).toEqual({
      kind: 'high-as-low',
    });
    expect(getWanExpertWiringWarning(tagged('i2v_a14b', 'low', 'm'), tagged('i2v_a14b', 'none', 'l'))).toEqual({
      kind: 'low-as-main',
    });
    expect(getWanExpertWiringWarning(tagged('i2v_a14b', 'low', 'm'), null)).toEqual({ kind: 'single-low' });
  });

  it('keeps warning after a role exchange of a same-tag pair, so no swap is offered', () => {
    // high+high and low+low pairs re-warn with roles exchanged — the UI uses
    // this simulation to withhold the Swap button rather than loop.
    const highPair = [tagged('i2v_a14b', 'high', 'm'), tagged('i2v_a14b', 'high', 'l')] as const;
    const lowPair = [tagged('i2v_a14b', 'low', 'm'), tagged('i2v_a14b', 'low', 'l')] as const;

    expect(getWanExpertWiringWarning(highPair[0], highPair[1])).toEqual({ kind: 'high-as-low' });
    expect(getWanExpertWiringWarning(highPair[1], highPair[0])).toEqual({ kind: 'high-as-low' });
    expect(getWanExpertWiringWarning(lowPair[0], lowPair[1])).toEqual({ kind: 'low-as-main' });
    expect(getWanExpertWiringWarning(lowPair[1], lowPair[0])).toEqual({ kind: 'low-as-main' });
    // Every swappable kind resolves when roles are exchanged.
    expect(getWanExpertWiringWarning(tagged('i2v_a14b', 'high', 'l'), tagged('i2v_a14b', 'low', 'm'))).toBeNull();
    expect(getWanExpertWiringWarning(tagged('i2v_a14b', 'high', 'l'), tagged('i2v_a14b', 'none', 'm'))).toBeNull();
    expect(getWanExpertWiringWarning(tagged('i2v_a14b', 'none', 'l'), tagged('i2v_a14b', 'low', 'm'))).toBeNull();
  });

  it('stays silent for correct, untagged, Diffusers, TI2V-5B, and non-Wan wirings', () => {
    expect(getWanExpertWiringWarning(tagged('i2v_a14b', 'high', 'm'), tagged('i2v_a14b', 'low', 'l'))).toBeNull();
    expect(getWanExpertWiringWarning(tagged('i2v_a14b', 'none', 'm'), tagged('i2v_a14b', 'none', 'l'))).toBeNull();
    expect(getWanExpertWiringWarning(tagged('i2v_a14b', 'high', 'm'), null)).toBeNull();
    expect(getWanExpertWiringWarning(wanModel('i2v_a14b', 'diffusers'), null)).toBeNull();
    expect(getWanExpertWiringWarning(tagged('ti2v_5b', 'low', 'm'), null)).toBeNull();
    expect(getWanExpertWiringWarning(h3Model(), null)).toBeNull();
    expect(getWanExpertWiringWarning(null, null)).toBeNull();
  });
});

describe('getVideoModelSelectionResult', () => {
  it('clears conditioning media the new model cannot consume', () => {
    const from = settingsFor(wanModel('i2v_a14b'), {
      firstFrameImage: FIRST_FRAME,
      lastFrameImage: LAST_FRAME,
      acceleratorEnabled: false,
    });
    const result = getVideoModelSelectionResult({
      currentSettings: from,
      model: wanModel('t2v_a14b'),
      models: [],
    });

    expect(result.settings.firstFrameImage).toBeNull();
    expect(result.settings.lastFrameImage).toBeNull();
    expect(result.clearedLabels).toContain('First frame');
    expect(result.clearedLabels).toContain('Last frame');
  });

  it('keeps a first+last pair on a model with FLF2V, but drops the last frame on TI2V-5B', () => {
    const from = settingsFor(wanModel('i2v_a14b'), { firstFrameImage: FIRST_FRAME, lastFrameImage: LAST_FRAME });

    const toH3 = getVideoModelSelectionResult({ currentSettings: from, model: h3Model(), models: [] });

    expect(toH3.settings.firstFrameImage).toEqual(FIRST_FRAME);
    expect(toH3.settings.lastFrameImage).toEqual(LAST_FRAME);

    const to5b = getVideoModelSelectionResult({ currentSettings: from, model: wanModel('ti2v_5b'), models: [] });

    expect(to5b.settings.firstFrameImage).toEqual(FIRST_FRAME);
    expect(to5b.settings.lastFrameImage).toBeNull();
  });

  it('drops a destination image for extend when the model lacks the end-frame channel', () => {
    const from = settingsFor(wanModel('i2v_a14b'), { lastFrameImage: LAST_FRAME, sourceVideo: SOURCE_VIDEO });
    const to5b = getVideoModelSelectionResult({ currentSettings: from, model: wanModel('ti2v_5b'), models: [] });

    expect(to5b.settings.sourceVideo).toEqual(SOURCE_VIDEO);
    expect(to5b.settings.lastFrameImage).toBeNull();
  });

  it('re-fits presets, frames, fps, Lightning, and components when crossing families', () => {
    const wan = wanModel('t2v_a14b');
    const catalog = [LIGHTNING_T2V_HIGH, LIGHTNING_T2V_LOW];
    const lightningOn = getAcceleratorToggleResult(settingsFor(wan), wan, catalog, true).settings;
    const from = settingsFor(wan, {
      ...lightningOn,
      loras: [...lightningOn.loras, { isEnabled: true, model: lora('Style LoRA') as never, weight: 0.5 }],
      targetResolution: '480p',
      vae: { base: 'wan', key: 'wan-vae', name: 'Wan VAE', type: 'vae' },
      wanT5EncoderModel: { base: 'any', key: 'umt5', name: 'UMT5-XXL', type: 'wan_t5_encoder' },
    });
    const result = getVideoModelSelectionResult({ currentSettings: from, model: h3Model(), models: catalog });

    expect(result.settings.targetResolution).toBe('768 highres');
    expect(result.settings.numFrames).toBe(90); // 81 snapped onto the H3 grid
    expect(result.settings.fps).toBe(24);
    expect(result.settings.acceleratorEnabled).toBe(false);
    expect(result.settings.steps).toBe(50);
    expect(result.settings.loras).toEqual([]); // wan LoRAs are incompatible with H3
    expect(result.settings.vae).toBeNull();
    expect(result.settings.wanT5EncoderModel).toBeNull();
    expect(result.clearedLabels).toEqual(
      expect.arrayContaining(['Target resolution', 'Frames', 'FPS', 'Acceleration', 'LoRAs', 'VAE', 'Wan T5 Encoder'])
    );
  });

  it('leaves user-tuned sampling and LoRA weights alone when the accelerator carries over unchanged', () => {
    const wan = wanModel('t2v_a14b');
    const catalog = [LIGHTNING_T2V_HIGH, LIGHTNING_T2V_LOW];
    const on = getAcceleratorToggleResult(settingsFor(wan), wan, catalog, true).settings;
    const tuned = {
      ...on,
      cfgScale: 2,
      loras: [
        { ...on.loras[0]!, weight: 0.5 },
        on.loras[1]!,
        { isEnabled: true, model: lora('Style LoRA') as never, weight: 0.7 },
      ],
      steps: 8,
    };
    const result = getVideoModelSelectionResult({
      currentSettings: tuned,
      model: wanModel('t2v_a14b', 'checkpoint'),
      models: catalog,
    });

    // Same accelerator LoRA set on the new model: nothing is re-applied.
    expect(result.settings).toMatchObject({ acceleratorEnabled: true, cfgScale: 2, steps: 8 });
    expect(result.settings.loras[0]?.weight).toBe(0.5);
    expect(result.settings.loras.map((entry) => entry.model.name)).toContain('Style LoRA');
    expect(result.clearedLabels).toEqual([]);
  });

  it('returns no cleared labels when everything carries over', () => {
    const model = wanModel('i2v_a14b');
    const from = settingsFor(model, { firstFrameImage: FIRST_FRAME, acceleratorEnabled: false });
    const result = getVideoModelSelectionResult({
      currentSettings: from,
      model: wanModel('i2v_a14b', 'diffusers'),
      models: [],
    });

    expect(result.clearedLabels).toEqual([]);
    expect(result.settings.modelKey).toBe(wanModel('i2v_a14b', 'diffusers').key);
  });
});

describe('getVideoValidationReasons', () => {
  it('blocks unsupported models', () => {
    expect(getVideoValidationReasons(h3Model('checkpoint'), settingsFor())).toEqual([
      'Video needs a supported video model before it can be invoked.',
    ]);
  });

  it('reports unsupported modes in plain language', () => {
    const t2v = wanModel('t2v_a14b', 'diffusers');
    const i2v = wanModel('i2v_a14b', 'diffusers');

    expect(getVideoValidationReasons(t2v, settingsFor(t2v, { firstFrameImage: FIRST_FRAME }))).toEqual([
      expect.stringContaining('does not support starting from a first frame'),
    ]);
    expect(getVideoValidationReasons(i2v, settingsFor(i2v))).toEqual([
      expect.stringContaining('does not support text-to-video'),
    ]);
  });

  it('rejects a destination image while extending on TI2V-5B', () => {
    const model = wanModel('ti2v_5b', 'diffusers');
    const settings = settingsFor(model, { lastFrameImage: LAST_FRAME, sourceVideo: SOURCE_VIDEO });

    expect(getVideoValidationReasons(model, settings)).toEqual([
      expect.stringContaining('cannot target a destination image'),
    ]);
  });

  it('rejects the first-frame + initial-video combination', () => {
    const model = wanModel('i2v_a14b', 'diffusers');
    const settings = settingsFor(model, { firstFrameImage: FIRST_FRAME, sourceVideo: SOURCE_VIDEO });

    expect(getVideoValidationReasons(model, settings)).toContainEqual(expect.stringContaining('cannot be combined'));
  });

  it('rejects a Wan extension whose source clip frame rate falls outside 1-120 fps', () => {
    const model = wanModel('i2v_a14b', 'diffusers');

    // wan_l2v/video_concat accept 1-120; an out-of-range clip would fail only
    // AFTER the denoise. A slow-mo clip and an unprobeable sub-1 fps rate both
    // block up front; an ordinary clip raises nothing.
    expect(
      getVideoValidationReasons(model, settingsFor(model, { sourceVideo: { ...SOURCE_VIDEO, fps: 240 } }))
    ).toContainEqual(expect.stringContaining('1-120 fps'));
    expect(
      getVideoValidationReasons(model, settingsFor(model, { sourceVideo: { ...SOURCE_VIDEO, fps: 0.3 } }))
    ).toContainEqual(expect.stringContaining('1-120 fps'));
    expect(getVideoValidationReasons(model, settingsFor(model, { sourceVideo: SOURCE_VIDEO }))).toEqual([]);
  });

  it('validates frames, fps, and steps against the matrix', () => {
    const wan = wanModel('t2v_a14b', 'diffusers');
    const h3 = h3Model();

    expect(getVideoValidationReasons(wan, settingsFor(wan, { numFrames: 80 }))).toEqual([
      expect.stringContaining('4·n + 1'),
    ]);
    expect(getVideoValidationReasons(h3, settingsFor(h3, { numFrames: 100 }))).toEqual([
      expect.stringContaining('17·n + 5'),
    ]);
    expect(getVideoValidationReasons(h3, settingsFor(h3, { fps: 16 }))).toEqual([
      expect.stringContaining('fixed 24 FPS'),
    ]);
    expect(getVideoValidationReasons(wan, settingsFor(wan, { fps: 0 }))).toEqual([
      expect.stringContaining('FPS must be a whole number between'),
    ]);
    expect(getVideoValidationReasons(h3, settingsFor(h3, { steps: 1 }))).toEqual([
      expect.stringContaining('at least 2'),
    ]);
  });

  it('rejects fractional fps and steps — the backend fields are integers', () => {
    const wan = wanModel('t2v_a14b', 'diffusers');

    expect(getVideoValidationReasons(wan, settingsFor(wan, { fps: 12.5 }))).toEqual([
      expect.stringContaining('whole number'),
    ]);
    expect(getVideoValidationReasons(wan, settingsFor(wan, { steps: 4.5 }))).toEqual([
      expect.stringContaining('whole number'),
    ]);
  });

  it('surfaces Wan LoRA family mismatches instead of silently dropping them', () => {
    const model = wanModel('ti2v_5b', 'diffusers');
    const settings = settingsFor(model, {
      loras: [{ isEnabled: true, model: LIGHTNING_T2V_HIGH as never, weight: 1 }],
    });

    expect(getVideoValidationReasons(model, settings)).toEqual([
      expect.stringContaining('targets a different Wan model family'),
    ]);
  });

  it('accepts a fully valid H3 first-frame setup', () => {
    const model = h3Model();

    expect(getVideoValidationReasons(model, settingsFor(model, { firstFrameImage: FIRST_FRAME }))).toEqual([]);
  });
});

describe('getVideoModelAvailabilityReasons', () => {
  it('reports uninstalled selections by label', () => {
    const model = wanModel('i2v_a14b');
    const settings = settingsFor(model, {
      loras: [{ isEnabled: true, model: LIGHTNING_T2V_HIGH as never, weight: 1 }],
      wanT5EncoderModel: { base: 'any', key: 'umt5', name: 'UMT5-XXL', type: 'wan_t5_encoder' },
    });

    const reasons = getVideoModelAvailabilityReasons(model, settings, []);

    expect(reasons).toContainEqual(expect.stringContaining('is no longer installed'));
    expect(reasons.some((reason) => reason.includes('UMT5-XXL'))).toBe(true);
    expect(reasons.some((reason) => reason.includes('Lightning'))).toBe(true);

    const installed = getVideoModelAvailabilityReasons(model, settingsFor(model), [model]);

    expect(installed).toEqual([]);
  });
});
