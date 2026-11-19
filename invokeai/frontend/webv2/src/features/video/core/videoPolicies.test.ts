import type { GenerationModelCatalogItem, MainModelConfig } from '@features/generation/contracts';

import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { LTX2_DEFAULT_NEGATIVE_PROMPT, LTX2_EXTEND_CONTEXT_FRAMES } from '@features/video/core/dimensions';
import { describe, expect, it } from 'vitest';

import type { VideoSettings } from './types';

import {
  findMiniMaxH3TurboLora,
  findWanLightningLoraPair,
  getAcceleratorToggleResult,
  getDefaultVideoSettings,
  getEffectiveVideoTiming,
  getVideoComponentSectionPolicy,
  getVideoDimensions,
  getVideoExpandPromptSuggestion,
  getVideoIssueSections,
  getVideoModelAvailabilityIssues,
  getVideoModelPolicy,
  getVideoPanelFields,
  getVideoModelSelectionResult,
  getVideoModes,
  getVideoPromptPolicy,
  getVideoValidationIssues,
  getVideoValidationReasons,
  getAutoDurationBounds,
  isAutoDurationSupportedForMode,
  getWanExpertWiringWarning,
  isSupportedVideoModel,
  isValidVideoNumFrames,
  isVideoModelSelectable,
  snapVideoNumFrames,
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

const lossKeys = (result: { losses: readonly { key: string }[] }) => result.losses.map((loss) => loss.key);

describe('isSupportedVideoModel', () => {
  it('accepts Wan mains in any format', () => {
    expect(isSupportedVideoModel(wanModel('t2v_a14b', 'gguf_quantized'))).toBe(true);
    expect(isSupportedVideoModel(wanModel('i2v_a14b', 'diffusers'))).toBe(true);
    expect(isSupportedVideoModel(wanModel('ti2v_5b', 'checkpoint'))).toBe(true);
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
});

describe('getVideoExpandPromptSuggestion', () => {
  const frame = { height: 512, image_name: 'frame.png', width: 768 };
  const promptSettings = {
    acceleratorEnabled: false,
    audioCfgScale: null,
    cfgScale: 3,
    cfgScaleLowNoise: null,
    negativePromptEnabled: true,
    wanLowNoiseModel: null,
  };

  it.each(['ltx2_dev', 'ltx2_distilled'])('suggests the LTX-2.5 enhancer and seeded prompts for %s', (variant) => {
    const { enhancer } = getVideoPromptPolicy(ltx2(variant), promptSettings);

    // The ids are the rows seeded by backend migration 2026_09_26_add_ltx2_system_prompts.
    expect(getVideoExpandPromptSuggestion(enhancer, frame)).toEqual({
      image: frame,
      imageSystemPromptId: '0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0010',
      modelName: 'LTX-2.5 Prompt Enhancer (Gemma-4 E2B)',
      modelSource: 'google/gemma-4-E2B-it',
      systemPromptId: '0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0009',
    });
    expect(getVideoExpandPromptSuggestion(enhancer, null)?.image).toBeNull();
  });
});

describe('getVideoPromptPolicy', () => {
  const promptSettings = (overrides: Partial<Parameters<typeof getVideoPromptPolicy>[1]> = {}) => ({
    acceleratorEnabled: false,
    audioCfgScale: null,
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
    expect(atCfg1.negativeHelpTextKey).toBe('widgets.video.negativeCfgHelp');
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
});

describe('LTX-2 extend context control', () => {
  const model = ltx2('ltx2_dev');
  const extendSettings = (overrides: Partial<VideoSettings> = {}) => ({
    ...getDefaultVideoSettings(model, []),
    sourceVideo: {
      endFrame: 200,
      fps: 24,
      height: 704,
      numFrames: 201,
      startFrame: 0,
      video_name: 's.mp4',
      width: 1248,
    },
    ...overrides,
  });

  it('bounds the control by what the join can afford for THIS source', () => {
    const hd = getVideoModelPolicy(model, extendSettings()).ui.extendContext;
    const uhd = getVideoModelPolicy(
      model,
      extendSettings({
        sourceVideo: {
          endFrame: 200,
          fps: 24,
          height: 2160,
          numFrames: 201,
          startFrame: 0,
          video_name: 's.mp4',
          width: 3840,
        },
      })
    ).ui.extendContext;

    expect(hd?.max ?? 0).toBeGreaterThan(LTX2_EXTEND_CONTEXT_FRAMES);
    expect(uhd).toMatchObject({ max: 0 });
  });

  it('refuses a context the source cannot afford, naming what would fit', () => {
    const reasons = getVideoValidationReasons(
      model,
      extendSettings({
        ltx2ExtendContextFrames: 97,
        sourceVideo: {
          endFrame: 200,
          fps: 24,
          height: 1440,
          numFrames: 201,
          startFrame: 0,
          video_name: 's.mp4',
          width: 2560,
        },
      })
    );

    expect(reasons.join(' ')).toMatch(/Context Frames is 97/);
  });
});

describe('LTX-2 distilled accelerator', () => {
  const DISTILLED = { base: 'ltx-2', key: 'ltx2-distilled', name: 'LTX-2.5 Distilled LoRA', type: 'lora' as const };
  it('turns the whole guided recipe off, not just the step count', () => {
    const model = ltx2('ltx2_dev');
    const settings = getDefaultVideoSettings(model, []);
    const result = getAcceleratorToggleResult(settings, model, [DISTILLED], true);

    expect(result.missingLoras).toBe(false);
    expect(result.settings).toMatchObject({
      acceleratorEnabled: true,
      acceleratorLoraKeys: ['ltx2-distilled'],
      audioCfgScale: 1,
      cfgScale: 1,
      modalityScale: 1,
      steps: 8,
      stgScale: 0,
    });
  });

  it('puts the guided recipe back when switched off', () => {
    const model = ltx2('ltx2_dev');
    const on = getAcceleratorToggleResult(getDefaultVideoSettings(model, []), model, [DISTILLED], true).settings;
    const off = getAcceleratorToggleResult(on, model, [DISTILLED], false).settings;

    expect(off).toMatchObject({
      acceleratorEnabled: false,
      acceleratorLoraKeys: [],
      audioCfgScale: 7,
      cfgScale: 3,
      modalityScale: 3,
      steps: 30,
      stgScale: 1,
    });
    expect(off.loras).toEqual([]);
  });
});

describe('MiniMax H3 Turbo', () => {
  const TURBO = { base: 'minimax-h3', key: 'turbo', name: 'MiniMax H3 Turbo LoRA', type: 'lora' as const };
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

    expect(off.loras.map((entry) => entry.model.key)).toEqual(['rider']);
    expect(off.acceleratorLoraKeys).toEqual([]);
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

  it('requires a component source (and a conditional text encoder) for a single-file H3 main', () => {
    const checkpoint = h3Model('checkpoint');
    const bare = settingsFor(checkpoint);
    const componentsOnly = { ...h3Model('diffusers', 'h3-components'), components_only: true };
    const encoder = { base: 'minimax-h3', key: 'h3-te-int8', name: 'H3 Text Encoder', type: 'qwen3_vl_encoder' };

    expect(getVideoComponentSectionPolicy(checkpoint, bare).defaultOpen).toBe(true);
    expect(getVideoComponentSectionPolicy(checkpoint, bare).slots.map((slot) => slot.key)).toEqual([
      'componentSourceModel',
      'h3TextEncoderModel',
    ]);

    expect(getVideoValidationReasons(checkpoint, bare)).toContainEqual(
      expect.stringContaining('select a Diffusers MiniMax H3 install under Model Components')
    );

    expect(getVideoValidationReasons(checkpoint, settingsFor(checkpoint, { componentSourceModel: h3Model() }))).toEqual(
      []
    );

    expect(
      getVideoValidationReasons(checkpoint, settingsFor(checkpoint, { componentSourceModel: componentsOnly }))
    ).toContainEqual(expect.stringContaining('select a single-file Text encoder'));
    expect(
      getVideoValidationReasons(
        checkpoint,
        settingsFor(checkpoint, { componentSourceModel: componentsOnly, h3TextEncoderModel: encoder })
      )
    ).toEqual([]);

    const slot = getVideoComponentSectionPolicy(checkpoint, bare).slots[0];
    const ctx = { model: checkpoint, selectedComponents: bare, settings: bare };

    expect(slot?.filter?.(h3Model(), ctx)).toBe(true);
    expect(slot?.filter?.(componentsOnly, ctx)).toBe(true);
    expect(slot?.filter?.(h3Model('checkpoint', 'other-ckpt'), ctx)).toBe(false);
    expect(slot?.filter?.(wanModel('i2v_a14b', 'diffusers'), ctx)).toBe(false);

    const full = h3Model();

    expect(getVideoValidationReasons(full, settingsFor(full))).toEqual([]);
    expect(getVideoComponentSectionPolicy(full, settingsFor(full)).defaultOpen).toBe(false);
  });

  it('agrees with the served Wan rows on which VAE width each variant takes', () => {
    const wanRows = architectureCapabilitiesFixture.filter((row) => row.base === 'wan');

    expect(wanRows.map((row) => row.variant)).toContain('ti2v_5b');

    for (const row of wanRows) {
      const model = wanModel(row.variant ?? 'i2v_a14b');
      const settings = settingsFor(model);
      const ctx = { model, selectedComponents: settings, settings };
      const slot = getVideoComponentSectionPolicy(model, settings).slots.find((s) => s.key === 'vae');

      for (const width of [16, 48]) {
        const vae = { base: 'wan', key: `vae${width}`, latent_channels: width, name: `Wan VAE ${width}`, type: 'vae' };
        const served = row.vae?.accepted.some((entry) => entry.base === 'wan' && entry.latent_channels === width);

        expect(slot?.filter?.(vae, ctx), `${row.variant ?? 'base row'}, ${width} channels`).toBe(served ?? false);
      }
    }
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
    expect(lossKeys(result)).toContain('firstFrame');
    expect(lossKeys(result)).toContain('lastFrame');
  });

  it('re-fits presets, frames, fps, Lightning, and components when crossing families', () => {
    const wan = wanModel('t2v_a14b');
    const catalog = [LIGHTNING_T2V_HIGH, LIGHTNING_T2V_LOW];
    const lightningOn = getAcceleratorToggleResult(settingsFor(wan), wan, catalog, true).settings;
    const from = settingsFor(wan, {
      ...lightningOn,
      loras: [...lightningOn.loras, { isEnabled: true, model: lora('Style LoRA') as never, weight: 0.5 }],
      targetResolution: '720p',
      vae: { base: 'wan', key: 'wan-vae', name: 'Wan VAE', type: 'vae' },
      wanT5EncoderModel: { base: 'any', key: 'umt5', name: 'UMT5-XXL', type: 'wan_t5_encoder' },
    });
    const result = getVideoModelSelectionResult({ currentSettings: from, model: h3Model(), models: catalog });

    expect(result.settings.targetResolution).toBe('768 highres');
    expect(result.settings.numFrames).toBe(90); // 81 snapped onto the H3 grid
    expect(result.settings.fps).toBe(24);
    expect(result.settings.acceleratorEnabled).toBe(false);
    expect(result.settings.steps).toBe(50);
    expect(result.settings.loras).toEqual([{ isEnabled: true, model: lora('Style LoRA'), weight: 0.5 }]);
    expect(result.settings.vae).toBeNull();
    expect(result.settings.wanT5EncoderModel).toBeNull();
    expect(result.adjustments).toEqual(expect.arrayContaining(['targetResolution', 'frames', 'fps', 'acceleration']));
    expect(result.losses).toEqual([
      { count: 1, key: 'vae' },
      { count: 1, key: 'wanT5EncoderModel' },
    ]);
  });
});

describe('getVideoValidationReasons', () => {
  it('blocks unsupported models', () => {
    const sdxl = { base: 'sdxl', format: 'checkpoint', key: 'sdxl', name: 'SDXL', type: 'main' } as MainModelConfig;

    expect(getVideoValidationReasons(sdxl, settingsFor())).toEqual([
      'Video needs a supported video model before it can be invoked.',
    ]);
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

  it('accepts a fully valid H3 first-frame setup', () => {
    const model = h3Model();

    expect(getVideoValidationReasons(model, settingsFor(model, { firstFrameImage: FIRST_FRAME }))).toEqual([]);
  });
});

describe('getVideoModelAvailabilityIssues', () => {
  const messagesOf = (issues: readonly { message: string }[]) => issues.map((issue) => issue.message);

  it('reports uninstalled selections by label', () => {
    const model = wanModel('i2v_a14b');
    const settings = settingsFor(model, {
      loras: [{ isEnabled: true, model: LIGHTNING_T2V_HIGH as never, weight: 1 }],
      wanT5EncoderModel: { base: 'any', key: 'umt5', name: 'UMT5-XXL', type: 'wan_t5_encoder' },
    });

    const reasons = messagesOf(getVideoModelAvailabilityIssues(model, settings, []));

    expect(reasons).toContainEqual(expect.stringContaining('is no longer installed'));
    expect(reasons.some((reason) => reason.includes('UMT5-XXL'))).toBe(true);
    expect(reasons.some((reason) => reason.includes('Lightning'))).toBe(true);

    const installed = messagesOf(getVideoModelAvailabilityIssues(model, settingsFor(model), [model]));

    expect(installed).toEqual([]);
  });
});

const ref2vaTransformer = (key = 'h3-ref2va-ckpt'): MainModelConfig => ({
  ...h3Model('checkpoint', key),
  name: 'MiniMax H3 Ref2VA Transformer (int8, pruned)',
  variant: 'ref2va',
});

const fl2vaTransformer = (key = 'h3-fl2va-ckpt'): MainModelConfig => ({
  ...h3Model('checkpoint', key),
  name: 'MiniMax H3 FL2VA Transformer (int8, pruned)',
  variant: 'fl2va',
});

const imageReference = { detail: 'max', image: { height: 4, image_name: 'ref.png', width: 4 }, kind: 'image' } as const;

describe('reference mode policy', () => {
  it('a ref2va transformer at top switches the panel to the reference-only mode set', () => {
    const model = ref2vaTransformer();
    const settings = settingsFor(model);

    expect(getVideoModelPolicy(model, settings).modes).toEqual(['reference']);
    expect(getVideoModelPolicy(model, settings).references).toEqual({ extend: true, maxImages: 9, maxVideos: 3 });
    expect(getVideoModelPolicy(h3Model(), settingsFor(h3Model())).references).toBeNull();
  });

  it('validates the reference rules', () => {
    const model = ref2vaTransformer();
    const withRefTransformer = (overrides: Partial<VideoSettings>) =>
      settingsFor(model, { componentSourceModel: h3Model(), ...overrides });

    expect(getVideoValidationReasons(model, withRefTransformer({}))).toContain(
      'Reference-to-video needs at least one reference.'
    );
    expect(
      getVideoValidationReasons(
        model,
        withRefTransformer({
          references: [
            {
              clip: {
                endFrame: 10,
                fps: 24,
                height: 4,
                numFrames: 12,
                startFrame: 0,
                video_name: 'ref.mp4',
                width: 4,
              },
              conditioning: 'audio',
              kind: 'video',
            },
          ],
        })
      )
    ).toEqual([]);
    expect(getVideoValidationReasons(model, withRefTransformer({ references: [imageReference] }))).toEqual([]);
  });
});

describe('H3 task switches through the top model selection', () => {
  it('clears frame media when switching to a ref2va transformer, and references on the way back', () => {
    const flModel = h3Model();
    const first = { height: 4, image_name: 'kf.png', width: 4 };
    const toRef = getVideoModelSelectionResult({
      currentSettings: settingsFor(flModel, { firstFrameImage: first, modelKey: flModel.key }),
      model: ref2vaTransformer(),
      models: [flModel],
    });

    expect(toRef.settings.firstFrameImage).toBeNull();
    expect(lossKeys(toRef)).toContain('firstFrame');

    const backToFl = getVideoModelSelectionResult({
      currentSettings: { ...toRef.settings, references: [imageReference] },
      model: fl2vaTransformer(),
      models: [flModel],
    });

    expect(backToFl.settings.references).toEqual([]);
    expect(lossKeys(backToFl)).toContain('references');
  });
});

describe('ref2va accelerator auto-pick', () => {
  const REF2V_TURBO = {
    base: 'minimax-h3',
    key: 'ref2v-turbo',
    name: 'MiniMax H3 Ref2V Turbo LoRA',
    type: 'lora' as const,
  };
  const FL2VA_TURBO = { base: 'minimax-h3', key: 'turbo', name: 'MiniMax H3 Turbo LoRA', type: 'lora' as const };

  it('requires the ref2v-token repack for the ref2va variant and excludes it otherwise', () => {
    expect(findMiniMaxH3TurboLora([REF2V_TURBO, FL2VA_TURBO], { variant: 'ref2va' })).toMatchObject({
      key: 'ref2v-turbo',
    });
    expect(findMiniMaxH3TurboLora([REF2V_TURBO, FL2VA_TURBO], { variant: 'fl2va' })).toMatchObject({ key: 'turbo' });
    expect(findMiniMaxH3TurboLora([FL2VA_TURBO], { variant: 'ref2va' })).toBeNull();
  });
});

describe('H3 component-source seeding', () => {
  it('model selection fills an empty component-source slot but keeps an explicit pick', () => {
    const checkpoint = h3Model('checkpoint');
    const componentsOnly = { ...h3Model('diffusers', 'h3-components'), components_only: true };
    const full = h3Model();
    const fromEmpty = getVideoModelSelectionResult({
      currentSettings: settingsFor(h3Model(), { modelKey: h3Model().key }),
      model: checkpoint,
      models: [full, componentsOnly],
    });

    expect(fromEmpty.settings.componentSourceModel?.key).toBe(full.key);

    const explicit = getVideoModelSelectionResult({
      currentSettings: settingsFor(checkpoint, { componentSourceModel: componentsOnly, modelKey: checkpoint.key }),
      model: checkpoint,
      models: [full, componentsOnly],
    });

    expect(explicit.settings.componentSourceModel?.key).toBe(componentsOnly.key);
  });
});

describe('reference-extend policy', () => {
  const initialVideo = {
    endFrame: 400,
    fps: 24,
    height: 480,
    numFrames: 402,
    startFrame: 0,
    video_name: 'long.mp4',
    width: 832,
  };

  it('accepts references alongside an initial video on ref2va, rejecting the pair elsewhere', () => {
    const ref2va = ref2vaTransformer();
    const combined = settingsFor(ref2va, {
      componentSourceModel: h3Model(),
      references: [imageReference],
      sourceVideo: initialVideo,
    });

    expect(getVideoValidationReasons(ref2va, combined)).toEqual([]);

    const fl2va = h3Model();
    const reasons = getVideoValidationReasons(
      fl2va,
      settingsFor(fl2va, { references: [imageReference], sourceVideo: initialVideo })
    );

    expect(reasons).toContainEqual(expect.stringContaining('cannot be combined with an initial video on this model'));
  });
});

const ltx2 = (variant: string, overrides: Partial<MainModelConfig> = {}): MainModelConfig => ({
  base: 'ltx-2',
  format: 'checkpoint',
  key: `ltx2-${variant}`,
  name: `LTX-2 ${variant}`,
  type: 'main',
  variant,
  ...overrides,
});

const LTX2_COMPONENTS = ltx2('ltx2_dev', {
  format: 'diffusers',
  key: 'ltx2-components',
  name: 'LTX-2.5 Components',
});
const LTX2_ENCODER = {
  base: 'ltx-2',
  key: 'gemma4',
  name: 'LTX-2.5 Text Encoder',
  type: 'gemma4_encoder' as const,
};

describe('LTX-2 policy', () => {
  it('runs a GGUF build exactly like the single-file checkpoint, components from the folder', () => {
    const checkpoint = ltx2('ltx2_distilled');
    const gguf = ltx2('ltx2_distilled', { format: 'gguf_quantized', key: 'ltx2-distilled-gguf' });
    const missingComponents = { ltx2TextEncoderModel: LTX2_ENCODER };
    const complete = { componentSourceModel: LTX2_COMPONENTS, ltx2TextEncoderModel: LTX2_ENCODER };

    expect(isSupportedVideoModel(gguf)).toBe(true);
    expect(isVideoModelSelectable(gguf)).toBe(true);
    expect(getDefaultVideoSettings(gguf, [LTX2_COMPONENTS, LTX2_ENCODER]).componentSourceModel?.key).toBe(
      LTX2_COMPONENTS.key
    );
    expect(getVideoValidationReasons(gguf, settingsFor(gguf, missingComponents))).toEqual(
      getVideoValidationReasons(checkpoint, settingsFor(checkpoint, missingComponents))
    );
    expect(getVideoValidationReasons(gguf, settingsFor(gguf, missingComponents))).not.toEqual([]);
    expect(getVideoValidationReasons(gguf, settingsFor(gguf, complete))).toEqual([]);
  });

  describe('whole-modality conditioning clips', () => {
    // 96 frames at 24 fps: four seconds, which snaps DOWN to 89 on the 8n + 1 grid.
    const CLIP = { fps: 24, height: 704, numFrames: 96, video_name: 'clip.mp4', width: 1248 };
    const model = ltx2('ltx2_dev');
    const conditioned = (role: 'audio' | 'video', overrides: Partial<VideoSettings> = {}) =>
      settingsFor(model, {
        componentSourceModel: LTX2_COMPONENTS,
        conditioningClip: { clip: CLIP, fpsKnown: true, role },
        ltx2TextEncoderModel: LTX2_ENCODER,
        ...overrides,
      });

    it('takes the length from the clip in either role, and the frame rate only from its picture', () => {
      const audio = getEffectiveVideoTiming(model, conditioned('audio', { fps: 30, numFrames: 121 }));
      const video = getEffectiveVideoTiming(model, conditioned('video', { fps: 30, numFrames: 121 }));

      // A generated picture runs at whatever rate was asked for, so the soundtrack's four seconds cover 120 frames
      // there -- 113 after the snap -- and 89 at the clip's own 24.
      expect(audio).toEqual({ fps: 30, fpsFromClip: false, numFrames: 113, numFramesFromClip: true });
      expect(video).toEqual({ fps: 24, fpsFromClip: true, numFrames: 89, numFramesFromClip: true });
    });

    it('refuses a clip beside an initial video, and its picture beside frames', () => {
      const frame = { height: 704, image_name: 'frame.png', width: 1248 };
      const source = {
        endFrame: 94,
        fps: 24,
        height: 704,
        numFrames: 96,
        startFrame: 0,
        video_name: 's.mp4',
        width: 1248,
      };

      expect(getVideoValidationReasons(model, conditioned('audio', { sourceVideo: source }))).toContain(
        'A conditioning clip cannot be combined with an initial video or references. Clear one side.'
      );
      for (const slot of ['firstFrameImage', 'lastFrameImage'] as const) {
        expect(getVideoValidationReasons(model, conditioned('video', { [slot]: frame })), slot).toContain(
          "A conditioning clip's picture cannot be combined with first/last frames. Clear one side."
        );
      }
    });
  });

  it('exposes the per-modality guidance controls on dev and none on distilled', () => {
    const dev = getVideoModelPolicy(ltx2('ltx2_dev'), getDefaultVideoSettings(ltx2('ltx2_dev')));

    expect(dev.ui.cfgVisible).toBe(true);
    expect([dev.ui.audioCfgVisible, dev.ui.stgVisible, dev.ui.modalityVisible]).toEqual([true, true, true]);
    expect(dev.ui.stepsEditable).toBe(true);
    expect(dev.ui.audioOutput).toBe(true);

    const distilled = getVideoModelPolicy(ltx2('ltx2_distilled'), getDefaultVideoSettings(ltx2('ltx2_distilled')));

    expect(distilled.ui.cfgVisible).toBe(false);
    expect([distilled.ui.audioCfgVisible, distilled.ui.stgVisible, distilled.ui.modalityVisible]).toEqual([
      false,
      false,
      false,
    ]);
    expect(distilled.ui.stepsEditable).toBe(false);
    expect(distilled.defaults.steps).toBe(8);
  });

  it("seeds the release's own negative prompt, which dev guides against", () => {
    const settings = getDefaultVideoSettings(ltx2('ltx2_dev'), [LTX2_COMPONENTS, LTX2_ENCODER]);

    expect(settings.negativePrompt).toBe(LTX2_DEFAULT_NEGATIVE_PROMPT);
    expect(settings.negativePromptEnabled).toBe(true);
    expect(getDefaultVideoSettings(wanModel('t2v_a14b')).negativePrompt).toBe('');
  });

  it('offers the two-stage presets and gives only dev a refine budget', () => {
    const dev = getVideoModelPolicy(ltx2('ltx2_dev'), getDefaultVideoSettings(ltx2('ltx2_dev')));
    const distilled = getVideoModelPolicy(ltx2('ltx2_distilled'), getDefaultVideoSettings(ltx2('ltx2_distilled')));

    for (const policy of [dev, distilled]) {
      const twoStage = policy.targetResolutions.filter((option) => option.stages === 2);

      expect(twoStage.map((option) => option.id)).toEqual(['1024p', '1536p']);
    }
    expect(dev.refineSteps).toBe(8);
    expect(distilled.refineSteps).toBeUndefined();
    expect(dev.targetResolutions.find((option) => option.id === dev.defaults.targetResolution)?.stages).toBeUndefined();
  });

  it('requires the Gemma-4 encoder on every LTX-2 model shape', () => {
    for (const model of [ltx2('ltx2_dev'), ltx2('ltx2_dev', { format: 'diffusers', key: 'ltx2-folder' })]) {
      const settings = { ...getDefaultVideoSettings(model), ltx2TextEncoderModel: null };

      expect(getVideoValidationReasons(model, { ...settings, componentSourceModel: LTX2_COMPONENTS })).toContain(
        'LTX-2 needs its Gemma-4 text encoder — no LTX-2 model carries one.'
      );
    }
  });

  it('snaps frame counts onto the 8n + 1 grid and says so when one is off it', () => {
    const model = ltx2('ltx2_dev');

    expect(snapVideoNumFrames(model, 121)).toBe(121);
    expect(snapVideoNumFrames(model, 100)).toBe(97);
    expect(isValidVideoNumFrames(model, 121)).toBe(true);
    expect(isValidVideoNumFrames(model, 120)).toBe(false);

    const settings = {
      ...getDefaultVideoSettings(model, [LTX2_COMPONENTS, LTX2_ENCODER]),
      numFrames: 120,
    };

    expect(getVideoValidationReasons(model, settings).join(' ')).toContain('8·n + 1');
  });

  it('rebuilds the guidance and step count when the model family changes', () => {
    const wanSettings = {
      ...getDefaultVideoSettings(wanModel('t2v_a14b')),
      audioCfgScale: null,
      steps: 40,
    };
    const model = ltx2('ltx2_dev');
    const toDev = getVideoModelSelectionResult({
      currentSettings: wanSettings,
      model,
      models: [LTX2_COMPONENTS, LTX2_ENCODER],
    });

    expect(toDev.settings.audioCfgScale).toBe(7);
    expect(toDev.settings.stgScale).toBe(1);
    expect(toDev.settings.modalityScale).toBe(3);
    expect(toDev.adjustments).not.toContain('advancedGuidance');

    const distilled = ltx2('ltx2_distilled');
    const toDistilled = getVideoModelSelectionResult({
      currentSettings: toDev.settings,
      model: distilled,
      models: [LTX2_COMPONENTS, LTX2_ENCODER],
    });

    expect(toDistilled.settings.audioCfgScale).toBeNull();
    expect(toDistilled.settings.stgScale).toBeNull();
    expect(toDistilled.settings.modalityScale).toBeNull();
    expect(toDistilled.settings.steps).toBe(8);
    expect(toDistilled.adjustments).toEqual(expect.arrayContaining(['steps', 'advancedGuidance']));
  });
});

describe('auto duration', () => {
  const HEAD = { base: 'ltx-2', key: 'head', name: 'Duration head', type: 'ltx2_duration_head' };
  const settings = (overrides: Partial<VideoSettings>): VideoSettings =>
    ({
      ...getDefaultVideoSettings(ltx2('ltx2_dev')),
      autoDuration: true,
      ltx2DurationHeadModel: HEAD,
      ...overrides,
    }) as VideoSettings;

  it('has a length to decide for a prompt-only or image-anchored run', () => {
    expect(isAutoDurationSupportedForMode(settings({}))).toBe(true);
    expect(
      isAutoDurationSupportedForMode(settings({ firstFrameImage: { height: 704, image_name: 'a.png', width: 1248 } }))
    ).toBe(true);
  });

  it('sizes a continuation at the source rate, with the context counted against the ceiling', () => {
    const extending = settings({
      fps: 24,
      ltx2ExtendContextFrames: 25,
      numFrames: 121,
      sourceVideo: {
        endFrame: 48,
        fps: 30,
        height: 704,
        numFrames: 49,
        startFrame: 0,
        video_name: 's.mp4',
        width: 1248,
      },
    } as Partial<VideoSettings>);

    expect(isAutoDurationSupportedForMode(extending)).toBe(true);
    // 25 held frames leave 121 - 24 = 97 for the prediction, at 30 fps; the total is back at 121.
    expect(getAutoDurationBounds(ltx2('ltx2_dev'), extending)).toEqual({
      contextFrames: 25,
      fps: 30,
      maxFrames: 121,
      maxNewFrames: 96,
      maxSeconds: 97 / 30,
      minSeconds: 1,
    });
  });
});

describe('what a model switch discards', () => {
  const videoReference = {
    clip: { ...SOURCE_VIDEO, endFrame: 40, startFrame: 8 },
    conditioning: 'video' as const,
    kind: 'video' as const,
  };

  it('counts the references a model without reference mode drops, but not an untouched initial-video anchor', () => {
    const ref2va = ref2vaTransformer();
    const fl2va = fl2vaTransformer();
    const supplied = settingsFor(ref2va, {
      modelKey: ref2va.key,
      references: [imageReference, videoReference, { ...imageReference, detail: 'match' }],
    });

    const result = getVideoModelSelectionResult({ currentSettings: supplied, model: fl2va, models: [ref2va, fl2va] });

    expect(result.settings.references).toEqual([]);
    expect(result.losses).toEqual([{ count: 3, key: 'references' }]);

    const anchored = getVideoModelSelectionResult({
      currentSettings: settingsFor(fl2va, { modelKey: fl2va.key, sourceVideo: SOURCE_VIDEO }),
      model: ref2va,
      models: [ref2va, fl2va],
    }).settings;
    const anchor = anchored.references[0];

    expect(anchor).toMatchObject({ fromSourceVideo: true });
    expect(getVideoModelSelectionResult({ currentSettings: anchored, model: fl2va, models: [] }).losses).toEqual([]);

    const trimmed = { ...anchored, references: [{ ...anchor!, trimOverridden: true }] };

    expect(getVideoModelSelectionResult({ currentSettings: trimmed, model: fl2va, models: [] }).losses).toEqual([
      { count: 1, key: 'references' },
    ]);

    const reconditioned = { ...anchored, references: [{ ...anchor!, conditioning: 'video' as const }] };

    expect(getVideoModelSelectionResult({ currentSettings: reconditioned, model: fl2va, models: [] }).losses).toEqual([
      { count: 1, key: 'references' },
    ]);
  });

  it("keeps the user's own LoRAs through a switch, reconciling only the accelerator pair", () => {
    const wan = wanModel('t2v_a14b');
    const catalog = [wan, LIGHTNING_T2V_HIGH, LIGHTNING_T2V_LOW];
    const on = getAcceleratorToggleResult(settingsFor(wan, { modelKey: wan.key }), wan, catalog, true).settings;
    const styled = {
      ...on,
      loras: [
        ...on.loras,
        { isEnabled: true, model: lora('Ink') as never, weight: 0.5 },
        { isEnabled: false, model: lora('Grain') as never, weight: 0.3 },
      ],
    };

    const result = getVideoModelSelectionResult({ currentSettings: styled, model: h3Model(), models: catalog });

    expect(result.settings.loras).toEqual(styled.loras.slice(-2));
    expect(result.losses).toEqual([]);
    expect(result.adjustments).toContain('acceleration');
  });
});

describe('the Video sections an issue is fixed from', () => {
  const t2v = wanModel('t2v_a14b', 'diffusers');
  const i2v = wanModel('i2v_a14b', 'diffusers');
  const ti2v = wanModel('ti2v_5b', 'diffusers');
  const sectionsOf = (model: MainModelConfig, settings: VideoSettings, message: string) => {
    const issue = getVideoValidationIssues(model, settings).find((candidate) => candidate.message.includes(message));

    return issue ? getVideoIssueSections(issue, getVideoPanelFields(model, settings)) : undefined;
  };

  it.each([
    [
      'a first frame beside an initial video, both shown',
      i2v,
      { firstFrameImage: FIRST_FRAME, sourceVideo: SOURCE_VIDEO },
      'A first frame and an initial video',
      ['frames', 'source'],
    ],
    [
      'a destination image while extending',
      ti2v,
      { lastFrameImage: LAST_FRAME, sourceVideo: SOURCE_VIDEO },
      'cannot target a destination image',
      ['source'],
    ],
    ['text-to-video on an image-to-video model', i2v, {}, 'does not support text-to-video', ['frames', 'source']],
    [
      'a first frame the model cannot start from',
      t2v,
      { firstFrameImage: FIRST_FRAME },
      'does not support starting from a first frame',
      [],
    ],
    ['an extension the model cannot run', t2v, { sourceVideo: SOURCE_VIDEO }, 'does not support extending a video', []],
    ['a frame count off the grid', t2v, { numFrames: 80 }, '4·n + 1', ['dimensions']],
    ['an editable frame rate out of range', t2v, { fps: 0 }, 'FPS must be a whole number', ['dimensions']],
    ['a fixed frame rate no field sets', h3Model(), { fps: 16 }, 'fixed 24 FPS', []],
    ['too few steps', h3Model(), { steps: 1 }, 'at least 2', ['render']],
    ['steps on a fixed schedule', ltx2('ltx2_distilled'), { steps: 0 }, 'Steps must be a whole number', []],
    ['a low-noise CFG under 1', t2v, { cfgScaleLowNoise: 0.5 }, 'CFG (Low Noise)', ['render']],
    [
      'an initial video at a frame rate Wan cannot extend',
      i2v,
      { sourceVideo: { ...SOURCE_VIDEO, fps: 240 } },
      '1-120 fps',
      ['source'],
    ],
  ] as const)('places %s', (_, model, overrides, message, sections) => {
    expect(sectionsOf(model, settingsFor(model, overrides as Partial<VideoSettings>), message)).toEqual(sections);
  });
});
