import type { GenerationModelCatalogItem, MainModelConfig } from '@features/generation/contracts';

import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import {
  isLtx2TwoStage,
  LTX2_DEFAULT_NEGATIVE_PROMPT,
  LTX2_EXTEND_CONTEXT_FRAMES,
} from '@features/video/core/dimensions';
import { isVideoTargetResolution, normalizeVideoSettings } from '@features/video/core/settings';
import { describe, expect, it } from 'vitest';

import type { Ltx2TargetResolution, VideoSettings } from './types';

import {
  findMiniMaxH3TurboLora,
  findWanLightningLoraPair,
  getAcceleratorLoraChangeResult,
  getAcceleratorSteps,
  getAcceleratorToggleResult,
  findLtx2DistilledLora,
  getDefaultVideoSettings,
  getEffectiveVideoTiming,
  getVideoComponentSectionPolicy,
  getVideoDimensions,
  getVideoExpandPromptSuggestion,
  getVideoModelAvailabilityReasons,
  getVideoModelPolicy,
  getVideoModelSelectionResult,
  getVideoModes,
  getVideoPromptPolicy,
  getVideoValidationReasons,
  getWanExpertWiringWarning,
  isSupportedVideoModel,
  isValidVideoNumFrames,
  isVideoModelSelectable,
  MINIMAX_H3_REF2V_TURBO_ACCELERATOR,
  snapVideoNumFrames,
  WAN_LIGHTNING_ACCELERATOR,
} from './videoPolicies';
import { syncVideoWidgetValuesWithModels } from './widgetValues';

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

  it('accepts MiniMax H3 mains in Diffusers and single-file checkpoint formats', () => {
    expect(isSupportedVideoModel(h3Model('diffusers'))).toBe(true);
    // The single-file transformer checkpoint IS the model identity now.
    expect(isSupportedVideoModel(h3Model('checkpoint'))).toBe(true);
    expect(isSupportedVideoModel({ ...h3Model('gguf_quantized'), format: 'gguf_quantized' })).toBe(false);
  });

  it('offers only runnable identity-bearing H3 models in the top selector', () => {
    // Only loadable transformers are selectable; components-only folders belong in components and Ref2VA folder
    // transformers are unsupported.
    expect(isVideoModelSelectable(h3Model('checkpoint'))).toBe(true);
    expect(isVideoModelSelectable({ ...h3Model('checkpoint'), variant: 'ref2va' })).toBe(true);
    expect(isVideoModelSelectable(h3Model('diffusers'))).toBe(true);
    expect(isVideoModelSelectable({ ...h3Model('diffusers'), components_only: true })).toBe(false);
    expect(isVideoModelSelectable({ ...h3Model('diffusers'), variant: 'ref2va' })).toBe(false);
    expect(isVideoModelSelectable(wanModel('ti2v_5b', 'checkpoint'))).toBe(true);
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

  it('suggests nothing for a family without an enhancer', () => {
    const { enhancer } = getVideoPromptPolicy(wanModel('t2v_a14b'), promptSettings);

    expect(getVideoExpandPromptSuggestion(enhancer, frame)).toBeNull();
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

  it('offers the control only in extend mode, with a source to size it against', () => {
    // The ceiling is a property of the SOURCE's pixels -- the join blends at the clip's own
    // resolution, not the generation canvas -- so without a clip there is nothing to bound.
    expect(getVideoModelPolicy(model, extendSettings()).ui.extendContext).not.toBeNull();
    expect(getVideoModelPolicy(model, getDefaultVideoSettings(model, [])).ui.extendContext).toBeNull();
    expect(getVideoModelPolicy(wanModel('i2v_a14b'), extendSettings()).ui.extendContext).toBeNull();
  });

  it('bounds the control by what the join can afford for THIS source', () => {
    // video_concat buffers the crossfade at the source's native resolution and refuses over
    // 512 MiB, so a larger clip affords a shorter context. Without a live bound the user could set
    // a value refused only at enqueue -- after both encodes and the transformer have run.
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
    // 4K cannot blend even the smallest usable context; 0 is the panel's "not extendable" signal.
    expect(uhd).toMatchObject({ max: 0 });
  });

  it('reports the new material left after the join consumes the context', () => {
    // output = source + numFrames - context, so every held frame costs a frame of new video. That
    // trade is invisible in Frames alone, which is why the control states it.
    const at17 = getVideoModelPolicy(model, extendSettings({ numFrames: 121 })).ui.extendContext;
    const at49 = getVideoModelPolicy(model, extendSettings({ ltx2ExtendContextFrames: 49, numFrames: 121 })).ui
      .extendContext;

    expect(at17).toMatchObject({ newFrames: 104 });
    expect(at49).toMatchObject({ newFrames: 72 });
  });

  it('refuses a context the source cannot afford, naming what would fit', () => {
    // Not reachable by dragging (the control is bounded), but a recalled or stored value can carry
    // a context made for a smaller source.
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

  it('refuses a trim that keeps fewer frames than the join will blend', () => {
    const reasons = getVideoValidationReasons(
      model,
      extendSettings({
        ltx2ExtendContextFrames: 49,
        sourceVideo: {
          endFrame: 20,
          fps: 24,
          height: 704,
          numFrames: 201,
          startFrame: 0,
          video_name: 's.mp4',
          width: 1248,
        },
      })
    );

    expect(reasons.join(' ')).toMatch(/keeps only 21/);
  });

  it('snaps a stored off-grid value onto the VAE grid', () => {
    // The node snaps a ragged request DOWN silently, so an unsnapped setting would leave the panel
    // showing a count the run did not use.
    expect(normalizeVideoSettings({ ...extendSettings(), ltx2ExtendContextFrames: 24 })).toMatchObject({
      ltx2ExtendContextFrames: 17,
    });
    expect(normalizeVideoSettings({ ...extendSettings(), ltx2ExtendContextFrames: 3 })).toMatchObject({
      ltx2ExtendContextFrames: 9,
    });
  });
});

describe('LTX-2 distilled accelerator', () => {
  const DISTILLED = { base: 'ltx-2', key: 'ltx2-distilled', name: 'LTX-2.5 Distilled LoRA', type: 'lora' as const };
  const STYLE = { base: 'ltx-2', key: 'ltx2-style', name: 'LTX-2 Painterly', type: 'lora' as const };

  it('finds the distilled LoRA and ignores other families and other LTX-2 LoRAs', () => {
    expect(findLtx2DistilledLora([STYLE, DISTILLED])).toMatchObject({ key: 'ltx2-distilled' });
    expect(findLtx2DistilledLora([STYLE])).toBeNull();
    // A distillation LoRA for a different architecture must not satisfy the LTX-2 slot.
    expect(findLtx2DistilledLora([{ base: 'wan', key: 'w', name: 'Wan Distilled', type: 'lora' as const }])).toBeNull();
  });

  it('turns the whole guided recipe off, not just the step count', () => {
    // The distillation retrains the model to predict the clean sample directly. Cutting steps to 8
    // while still paying for CFG, STG and modality guidance samples off the distribution it was
    // fitted to -- the failure looks like a broken model, not like a slow one.
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

  it('hides the negative prompt once guidance is gone, like the distilled checkpoint', () => {
    // The accelerator drives every scale to identity, so the negative prompt stops being used --
    // but LTX-2 is the only family that pre-fills it, so leaving it on screen shows a populated box
    // full of terms that silently do nothing, under a CFG of 1 the user never typed.
    const model = ltx2('ltx2_dev');
    const guided = getVideoPromptPolicy(model, {
      acceleratorEnabled: false,
      audioCfgScale: 7,
      cfgScale: 3,
      cfgScaleLowNoise: null,
      negativePromptEnabled: true,
      wanLowNoiseModel: null,
    });
    const accelerated = getVideoPromptPolicy(model, {
      acceleratorEnabled: true,
      audioCfgScale: 1,
      cfgScale: 1,
      cfgScaleLowNoise: null,
      negativePromptEnabled: true,
      wanLowNoiseModel: null,
    });

    expect(guided).toMatchObject({ negativeUsedInGraph: true, negativeVisible: true });
    expect(accelerated).toMatchObject({ negativeUsedInGraph: false, negativeVisible: false });
    // The same shape the distilled checkpoint presents, which is the model an accelerated Dev is.
    expect(accelerated.negativeVisible).toBe(
      getVideoPromptPolicy(ltx2('ltx2_distilled'), {
        acceleratorEnabled: false,
        audioCfgScale: null,
        cfgScale: 1,
        cfgScaleLowNoise: null,
        negativePromptEnabled: true,
        wanLowNoiseModel: null,
      }).negativeVisible
    );
  });

  it('stops offering every control the distilled path would ignore', () => {
    // The graph sends `schedule: 'distilled'` whenever the accelerator is on, and the backend then
    // DISCARDS the guidance scales and the step count (`_resolve_guidance` and the step clamp both
    // log that they are ignoring what was asked). Leaving those scrubbers editable would let a user
    // set a value, see it accepted, and get a run that silently used something else.
    const model = ltx2('ltx2_dev');
    const guided = getVideoModelPolicy(model, getDefaultVideoSettings(model, []));
    const accelerated = getVideoModelPolicy(model, {
      ...getDefaultVideoSettings(model, []),
      acceleratorEnabled: true,
    });

    expect(guided.ui).toMatchObject({
      audioCfgVisible: true,
      cfgVisible: true,
      modalityVisible: true,
      stepsEditable: true,
      stgVisible: true,
    });
    expect(accelerated.ui).toMatchObject({
      audioCfgVisible: false,
      cfgVisible: false,
      modalityVisible: false,
      stepsEditable: false,
      stgVisible: false,
    });
    // Same presentation the distilled checkpoint gives, which is the model this now is.
    const checkpoint = getVideoModelPolicy(ltx2('ltx2_distilled'), getDefaultVideoSettings(ltx2('ltx2_distilled')));

    expect(accelerated.ui.stepsEditable).toBe(checkpoint.ui.stepsEditable);
    expect(accelerated.ui.cfgVisible).toBe(checkpoint.ui.cfgVisible);
    expect(accelerated.prompt.negativeVisible).toBe(checkpoint.prompt.negativeVisible);
  });

  it('restores the whole recipe when the accelerator LoRA is disabled, not just steps and CFG', () => {
    // Turning the accelerator off by unticking its LoRA in Concepts goes through a different path
    // than the toggle. If that path restores only steps and CFG, the run is the undistilled Dev
    // model with audio CFG, modality and STG still pinned at the accelerator's identity values --
    // guided sampling with three quarters of its guidance off, and nothing says so.
    const model = ltx2('ltx2_dev');
    const on = getAcceleratorToggleResult(getDefaultVideoSettings(model, []), model, [DISTILLED], true).settings;
    const disabled = on.loras.map((entry) => ({ ...entry, isEnabled: false }));
    const result = getAcceleratorLoraChangeResult({ ...on, loras: disabled }, model, [DISTILLED], disabled);

    expect(result.outcome).toBe('disabled');
    expect(result.settings).toMatchObject({
      acceleratorEnabled: false,
      audioCfgScale: 7,
      cfgScale: 3,
      modalityScale: 3,
      steps: 30,
      stgScale: 1,
    });
  });

  it('treats the negative prompt as unused even if a stale scale says otherwise', () => {
    // The panel cannot produce this, but recall can: metadata carrying a guided CFG alongside the
    // accelerator's LoRA set. The graph still sends `schedule: 'distilled'` whenever the toggle is
    // on, and the backend discards the scale -- so the answer must follow the accelerator, not the
    // stale number, or the panel claims a prompt is in use that the run throws away.
    const policy = getVideoPromptPolicy(ltx2('ltx2_dev'), {
      acceleratorEnabled: true,
      audioCfgScale: 7,
      cfgScale: 3,
      cfgScaleLowNoise: null,
      negativePromptEnabled: true,
      wanLowNoiseModel: null,
    });

    expect(policy).toMatchObject({ negativeUsedInGraph: false, negativeVisible: false });
  });

  it('writes the guidance recipe when the accelerator LoRA is swapped for another', () => {
    // The recorded LoRA is gone and a replacement is found, so the accelerator stays on under a
    // different file. That path has to write the same recipe the toggle does, or the swap silently
    // leaves guidance wherever it happened to be.
    const model = ltx2('ltx2_dev');
    const settings = {
      ...getDefaultVideoSettings(model, []),
      acceleratorEnabled: true,
      acceleratorLoraKeys: ['a-release-that-is-gone'],
      audioCfgScale: 7,
      cfgScale: 3,
      modalityScale: 3,
      stgScale: 1,
    };
    // The distilled LoRA is present and on; the key the toggle recorded is not.
    const present = [{ isEnabled: true, model: DISTILLED, weight: 1 }];
    const result = getAcceleratorLoraChangeResult({ ...settings, loras: present }, model, [DISTILLED], present);

    expect(result.outcome).toBe('switched');
    expect(result.settings).toMatchObject({
      acceleratorEnabled: true,
      audioCfgScale: 1,
      cfgScale: 1,
      modalityScale: 1,
      steps: 8,
      stgScale: 0,
    });
  });

  it('restores the whole recipe when the accelerator LoRA leaves the catalog entirely', () => {
    // A different route than unticking it in Concepts: this one runs when the LoRA is deleted in
    // Model Manager. It goes through `syncVideoWidgetValues`, which copies named fields out of the
    // change result -- so a field the result restores but the copy does not name is silently lost,
    // leaving a guided Dev run with its audio, STG and modality guidance pinned at identity.
    const model = ltx2('ltx2_dev');
    const on = getAcceleratorToggleResult(getDefaultVideoSettings(model, []), model, [DISTILLED], true).settings;
    // The LoRA is gone from the catalog: only the main model remains.
    const synced = syncVideoWidgetValuesWithModels({ ...on, loras: [], model }, [model]);

    expect(synced).toMatchObject({
      acceleratorEnabled: false,
      audioCfgScale: 7,
      cfgScale: 3,
      modalityScale: 3,
      steps: 30,
      stgScale: 1,
    });
  });

  it('leaves families whose accelerator keeps guidance alone', () => {
    // Wan and H3 declare no guidance triple, so their accelerator drops CFG but does not make the
    // run guidance-free -- their negative prompt stays visible, exactly as before.
    const wan = getVideoPromptPolicy(wanModel('t2v_a14b'), {
      acceleratorEnabled: true,
      audioCfgScale: null,
      cfgScale: 1,
      cfgScaleLowNoise: null,
      negativePromptEnabled: true,
      wanLowNoiseModel: null,
    });

    expect(wan.negativeVisible).toBe(true);
  });

  it('is not offered on the distilled checkpoint, which already is the fast path', () => {
    // Patching a distillation LoRA onto a model the distillation was not fitted to.
    const policy = getVideoModelPolicy(ltx2('ltx2_distilled'), getDefaultVideoSettings(ltx2('ltx2_distilled')));

    expect(policy.ui.accelerator).toBeNull();
  });
});

describe('MiniMax H3 Turbo', () => {
  const TURBO = { base: 'minimax-h3', key: 'turbo', name: 'MiniMax H3 Turbo LoRA', type: 'lora' as const };
  // Identify the alternative Turbo schedule from its organization naming.
  const LIGHTX2V = {
    base: 'minimax-h3',
    key: 'lightx2v',
    name: 'MiniMax H3 LightX2V Turbo LoRA',
    type: 'lora' as const,
  };
  const H3_CATALOG = [TURBO, LIGHTX2V];

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

  it('never auto-picks a Ref2VA-trained turbo LoRA for FL2VA generation', () => {
    // Ref2V Turbo is incompatible with FL2VA despite matching broad H3/turbo name patterns.
    const ref2vTurbo = { base: 'minimax-h3', key: 'ref2v', name: 'MiniMax H3 Ref2V Turbo LoRA', type: 'lora' as const };
    const ref2vFile = {
      base: 'minimax-h3',
      key: 'ref2v-file',
      name: 'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16',
      type: 'lora' as const,
    };

    expect(findMiniMaxH3TurboLora([ref2vTurbo, TURBO])).toMatchObject({ key: 'turbo' });
    expect(findMiniMaxH3TurboLora([ref2vTurbo])).toBeNull();
    expect(findMiniMaxH3TurboLora([ref2vFile])).toBeNull();
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

  it('runs the LightX2V release at the 8 steps it was distilled for, not the repack default of 6', () => {
    const model = h3Model();
    const on = getAcceleratorToggleResult(settingsFor(model), model, [LIGHTX2V], true).settings;

    expect(on).toMatchObject({ acceleratorEnabled: true, steps: 8 });
    expect(on.acceleratorLoraKeys).toEqual([LIGHTX2V.key]);
    // Help text reflects active LoRA steps while family accelerator policy remains reusable.
    expect(getVideoModelPolicy(model, on).ui).toMatchObject({ acceleratorSteps: 8 });
    expect(getVideoModelPolicy(model, on).ui.accelerator).toMatchObject({ label: 'Turbo', steps: 6 });
  });

  it('re-anchors the fast path on the other Turbo LoRA instead of tearing it down', () => {
    const model = h3Model();
    const on = getAcceleratorToggleResult(settingsFor(model), model, H3_CATALOG, true).settings;

    expect(on.acceleratorLoraKeys).toEqual([LIGHTX2V.key]);

    const swapped = getAcceleratorLoraChangeResult(on, model, H3_CATALOG, [
      { isEnabled: false, model: LIGHTX2V as never, weight: 1 },
      { isEnabled: true, model: TURBO as never, weight: 1 },
    ]);

    expect(swapped.outcome).toBe('switched');
    expect(swapped.settings).toMatchObject({ acceleratorEnabled: true, cfgScale: 1, steps: 6 });
    expect(swapped.settings.acceleratorLoraKeys).toEqual([TURBO.key]);
    expect(swapped.acceleratorLoras?.map((entry) => entry.key)).toEqual([TURBO.key]);
  });

  it('never arms an off fast path from a list edit, whatever lands in the list', () => {
    const model = h3Model();
    const turboRider = { base: 'minimax-h3', key: 'rider', name: 'Turbo Rider', type: 'lora' as const };
    // Preserve hand-tuned steps for ordinary user LoRAs that merely match the family name.
    const own = { base: 'minimax-h3', key: 'mine', name: 'My H3 Turbo Sharpener', type: 'lora' as const };
    const off = settingsFor(model, { steps: 12 });

    for (const candidate of [TURBO, LIGHTX2V, turboRider, own]) {
      const result = getAcceleratorLoraChangeResult(off, model, H3_CATALOG, [
        { isEnabled: true, model: candidate as never, weight: 1 },
      ]);

      expect(result.outcome).toBe('unchanged');
      expect(result.settings).toMatchObject({ acceleratorEnabled: false, steps: 12 });
    }
  });

  it('installs the catalog Turbo LoRA on toggle-on, never a user LoRA that is merely H3-named', () => {
    const model = h3Model();
    const own = { base: 'minimax-h3', key: 'mine', name: 'My H3 Turbo Sharpener', type: 'lora' as const };
    const withOwn = settingsFor(model, { loras: [{ isEnabled: true, model: own as never, weight: 0.4 }] });
    const on = getAcceleratorToggleResult(withOwn, model, [TURBO, own], true).settings;

    expect(on.acceleratorLoraKeys).toEqual([TURBO.key]);
    expect(on.steps).toBe(6);
    // ...and the user's own entry keeps its weight rather than being co-opted.
    expect(on.loras.find((entry) => entry.model.key === 'mine')?.weight).toBe(0.4);
  });

  it('keeps an oddly named Turbo LoRA running while it is the only one installed', () => {
    const model = h3Model();
    const turboRider = { base: 'minimax-h3', key: 'rider', name: 'Turbo Rider', type: 'lora' as const };
    const on = getDefaultVideoSettings(model, [turboRider]);

    expect(on.acceleratorLoraKeys).toEqual(['rider']);

    // An unrelated list edit must not tear down a fast path that is still on.
    const result = getAcceleratorLoraChangeResult(
      on,
      model,
      [turboRider],
      [...on.loras, { isEnabled: true, model: lora('Style LoRA') as never, weight: 0.5 }]
    );

    expect(result.outcome).toBe('unchanged');
    expect(result.settings.acceleratorEnabled).toBe(true);
  });

  it('turns the fast path off with the model defaults when no Turbo LoRA is left enabled', () => {
    const model = h3Model();
    const on = getDefaultVideoSettings(model, [TURBO]);
    const result = getAcceleratorLoraChangeResult(on, model, H3_CATALOG, []);

    expect(result.outcome).toBe('disabled');
    expect(result.settings).toMatchObject({ acceleratorEnabled: false, steps: 50 });
  });

  it('leaves user-tuned steps alone while its own LoRA is still enabled', () => {
    const model = h3Model();
    const on = { ...getDefaultVideoSettings(model, [TURBO]), steps: 9 };
    const result = getAcceleratorLoraChangeResult(on, model, H3_CATALOG, [
      ...on.loras,
      { isEnabled: true, model: lora('Style LoRA') as never, weight: 0.5 },
    ]);

    expect(result.outcome).toBe('unchanged');
    expect(result.settings).toMatchObject({ acceleratorEnabled: true, steps: 9 });
  });

  it('stays intact through a duplicated LoRA entry, so an unchanged value reconciles to itself', () => {
    const model = h3Model();
    const on = getDefaultVideoSettings(model, [TURBO]);
    // Duplicate persisted LoRA keys must not falsely break accelerator detection.
    const duplicated = [...on.loras, ...on.loras];
    const once = getAcceleratorLoraChangeResult(on, model, [TURBO], duplicated);
    const twice = getAcceleratorLoraChangeResult(once.settings, model, [TURBO], duplicated);

    expect(once.outcome).toBe('unchanged');
    expect(twice.outcome).toBe('unchanged');
    expect(twice.settings.acceleratorLoraKeys).toBe(on.acceleratorLoraKeys);
  });

  it('rejects a Lightning pair that does not name the family, even with no family token to check', () => {
    // Without a variant token, reject arbitrary named Lightning pairs rather than treating fallback as a wildcard.
    const fallbackMain = wanModel('', 'gguf_quantized', 'wan-unprobed');
    const mine = [
      lora('Personal Lightning High Detail', null, 'myh'),
      lora('Personal Lightning Low Detail', null, 'myl'),
    ];

    expect(findWanLightningLoraPair(mine, fallbackMain.variant, { requireFamilyName: true })).toBeNull();
    // Unrestricted (the catalog question) it still resolves, as before.
    expect(findWanLightningLoraPair(mine, fallbackMain.variant)).toMatchObject({ high: { key: 'myh' } });
  });

  it('swaps a T2V Lightning pair for the I2V one when the main changes family', () => {
    const catalog = [LIGHTNING_T2V_HIGH, LIGHTNING_T2V_LOW, LIGHTNING_I2V_HIGH, LIGHTNING_I2V_LOW];
    const t2v = wanModel('t2v_a14b');
    const on = getAcceleratorToggleResult(settingsFor(t2v), t2v, catalog, true).settings;
    const result = getVideoModelSelectionResult({
      currentSettings: on,
      model: wanModel('i2v_a14b'),
      models: catalog,
    });

    expect(result.settings.acceleratorLoraKeys).toEqual([LIGHTNING_I2V_HIGH.key, LIGHTNING_I2V_LOW.key]);
    expect(result.clearedLabels).toContain('Acceleration');
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

  it('requires a component source (and a conditional text encoder) for a single-file H3 main', () => {
    // Standalone H3 needs Diffusers components; components-only sources additionally require a separate text
    // encoder.
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

    // A full Diffusers source alone satisfies both slots.
    expect(getVideoValidationReasons(checkpoint, settingsFor(checkpoint, { componentSourceModel: h3Model() }))).toEqual(
      []
    );

    // A components-only source still needs the single-file encoder.
    expect(
      getVideoValidationReasons(checkpoint, settingsFor(checkpoint, { componentSourceModel: componentsOnly }))
    ).toContainEqual(expect.stringContaining('select a single-file Text encoder'));
    expect(
      getVideoValidationReasons(
        checkpoint,
        settingsFor(checkpoint, { componentSourceModel: componentsOnly, h3TextEncoderModel: encoder })
      )
    ).toEqual([]);

    // The source slot lists only Diffusers H3 installs.
    const slot = getVideoComponentSectionPolicy(checkpoint, bare).slots[0];
    const ctx = { model: checkpoint, selectedComponents: bare, settings: bare };

    expect(slot?.filter?.(h3Model(), ctx)).toBe(true);
    expect(slot?.filter?.(componentsOnly, ctx)).toBe(true);
    expect(slot?.filter?.(h3Model('checkpoint', 'other-ckpt'), ctx)).toBe(false);
    expect(slot?.filter?.(wanModel('i2v_a14b', 'diffusers'), ctx)).toBe(false);

    // A full Diffusers install at top needs nothing.
    const full = h3Model();

    expect(getVideoValidationReasons(full, settingsFor(full))).toEqual([]);
    expect(getVideoComponentSectionPolicy(full, settingsFor(full)).defaultOpen).toBe(false);
  });

  it('offers the hybrid quality base only on a Ref2VA checkpoint main, listing same-kind FL2VA checkpoints', () => {
    const ref2va: MainModelConfig = { ...h3Model('checkpoint', 'h3-ref2va'), pruned: true, variant: 'ref2va' };
    const settings = settingsFor(ref2va);
    const policy = getVideoComponentSectionPolicy(ref2va, settings);

    // Last: optional tuning sits below the slots the panel needs to run.
    expect(policy.slots.map((slot) => slot.key)).toEqual([
      'componentSourceModel',
      'h3TextEncoderModel',
      'h3HybridBaseModel',
    ]);

    const slot = policy.slots.find((candidate) => candidate.key === 'h3HybridBaseModel');
    const ctx = { model: ref2va, selectedComponents: settings, settings };

    // Optional: the panel is complete without it.
    expect(slot?.required).toBeUndefined();
    expect(
      getVideoValidationReasons(ref2va, settingsFor(ref2va, { componentSourceModel: h3Model() }))
    ).not.toContainEqual(expect.stringContaining('Hybrid'));

    // Hybrid choices must match pruned/full shape and exclude Ref2VA, Diffusers, and the selected main.
    expect(slot?.filter?.({ ...h3Model('checkpoint', 'fl2va-pruned'), pruned: true }, ctx)).toBe(true);
    expect(slot?.filter?.(h3Model('checkpoint', 'fl2va-kind-unknown'), ctx)).toBe(true);
    expect(slot?.filter?.({ ...h3Model('checkpoint', 'fl2va-full'), pruned: false }, ctx)).toBe(false);
    expect(slot?.filter?.({ ...h3Model('checkpoint', 'other-ref2va'), pruned: true, variant: 'ref2va' }, ctx)).toBe(
      false
    );
    expect(slot?.filter?.(ref2va, ctx)).toBe(false);
    expect(slot?.filter?.(h3Model('diffusers'), ctx)).toBe(false);

    // An FL2VA main has nothing to hybridize.
    const fl2va = h3Model('checkpoint');

    expect(getVideoComponentSectionPolicy(fl2va, settingsFor(fl2va)).slots.map((slot) => slot.key)).not.toContain(
      'h3HybridBaseModel'
    );
  });

  it('steers a legacy components-only H3 main at top toward the checkpoint-as-model shape', () => {
    const componentsOnly = { ...h3Model(), components_only: true };

    expect(getVideoValidationReasons(componentsOnly, settingsFor(componentsOnly))).toEqual([
      expect.stringContaining('Select a single-file MiniMax H3 transformer as the model'),
    ]);
    // The Ref2VA folder install cannot folder-load its transformer weights.
    const ref2vaFolder = { ...h3Model(), variant: 'ref2va' };

    expect(getVideoValidationReasons(ref2vaFolder, settingsFor(ref2vaFolder))).toEqual([
      expect.stringContaining('Select a single-file Ref2VA transformer as the model'),
    ]);
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

  it('agrees with the served Wan rows on which VAE width each variant takes', () => {
    // Assert backend VAE rules independently because stored VAE synchronization runs before capability rows
    // arrive.
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

  it('a cross-family component source covers the encoder but not the VAE', () => {
    // TI2V-5B main with an A14B Diffusers source: UMT5 is shared (encoder
    // satisfied) but the VAE is family-bound, so a VAE is still required.
    const model = wanModel('ti2v_5b', 'checkpoint');
    const settings = settingsFor(model, { componentSourceModel: wanModel('i2v_a14b', 'diffusers') });
    const reasons = getVideoValidationReasons(model, settings);

    expect(reasons).toContain('Video needs a VAE for Wan models.');
    expect(reasons).not.toContain('Video needs a Wan T5 Encoder for Wan models.');
  });

  it('offers only the optional text-encoder override for a full Diffusers H3 main', () => {
    const model = h3Model();
    const policy = getVideoComponentSectionPolicy(model, settingsFor(model));

    expect(policy.defaultOpen).toBe(false);
    expect(policy.slots.map((slot) => slot.key)).toEqual(['h3TextEncoderModel']);
    expect(policy.slots.every((slot) => !slot.required)).toBe(true);
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
    // Offer Swap only if swapping resolves warnings rather than repeating identical expert-role mismatches.
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

  it('drops the hybrid quality base when the new main is not a Ref2VA checkpoint, keeps it across Ref2VA files', () => {
    const ref2va: MainModelConfig = { ...h3Model('checkpoint', 'h3-ref2va'), pruned: true, variant: 'ref2va' };
    const fl2vaBase: MainModelConfig = { ...h3Model('checkpoint', 'h3-fl2va-base'), pruned: true };
    const from = settingsFor(ref2va, {
      componentSourceModel: h3Model(),
      h3HybridBaseModel: fl2vaBase,
      h3HybridStartBlock: 30,
    });

    const toFl2va = getVideoModelSelectionResult({ currentSettings: from, model: h3Model('checkpoint'), models: [] });

    expect(toFl2va.settings.h3HybridBaseModel).toBeNull();
    expect(toFl2va.clearedLabels).toContain('Hybrid quality base');

    const otherRef2va: MainModelConfig = { ...ref2va, key: 'h3-ref2va-2' };
    const toRef2va = getVideoModelSelectionResult({ currentSettings: from, model: otherRef2va, models: [] });

    expect(toRef2va.settings.h3HybridBaseModel).toEqual(fl2vaBase);
    expect(toRef2va.settings.h3HybridStartBlock).toBe(30);
    expect(toRef2va.clearedLabels).not.toContain('Hybrid quality base');
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
    const sdxl = { base: 'sdxl', format: 'checkpoint', key: 'sdxl', name: 'SDXL', type: 'main' } as MainModelConfig;

    expect(getVideoValidationReasons(sdxl, settingsFor())).toEqual([
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

  it('floors the low-noise CFG at 1 and accepts reuse-primary', () => {
    const model = wanModel('t2v_a14b', 'diffusers');
    const reason = 'CFG (Low Noise) must be at least 1.';

    expect(getVideoValidationReasons(model, settingsFor(model, { cfgScaleLowNoise: 0.5 }))).toContainEqual(reason);
    expect(getVideoValidationReasons(model, settingsFor(model, { cfgScaleLowNoise: 1 }))).not.toContainEqual(reason);
    expect(getVideoValidationReasons(model, settingsFor(model, { cfgScaleLowNoise: null }))).not.toContainEqual(reason);
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

    // Validate inherited Wan fps before denoising so invalid rates cannot fail only at video output.
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

  it('an unknown variant still falls back to fl2va, but ref2va never does', () => {
    const unknown = { ...h3Model(), variant: 'novel_task' };

    expect(getVideoModes(unknown)).toContain('txt2vid');
    expect(getVideoModes({ ...h3Model(), variant: 'ref2va' })).toEqual(['reference']);
  });

  it('validates the reference rules', () => {
    const model = ref2vaTransformer();
    // A full Diffusers source keeps the component slots satisfied, so only
    // the reference rules under test can produce reasons.
    const withRefTransformer = (overrides: Partial<VideoSettings>) =>
      settingsFor(model, { componentSourceModel: h3Model(), ...overrides });

    expect(getVideoValidationReasons(model, withRefTransformer({}))).toContain(
      'Reference-to-video needs at least one image or video reference.'
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
    ).toContain(
      'At least one reference must contribute visuals — add an image, or set a video reference to include video.'
    );
    expect(getVideoValidationReasons(model, withRefTransformer({ references: [imageReference] }))).toEqual([]);
  });

  it('rejects references on an fl2va panel', () => {
    const model = h3Model();
    const reasons = getVideoValidationReasons(model, settingsFor(model, { references: [imageReference] }));

    expect(reasons.some((reason) => reason.includes('reference-conditioned'))).toBe(true);
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
    expect(toRef.clearedLabels).toContain('First frame');

    const backToFl = getVideoModelSelectionResult({
      currentSettings: { ...toRef.settings, references: [imageReference] },
      model: fl2vaTransformer(),
      models: [flModel],
    });

    expect(backToFl.settings.references).toEqual([]);
    expect(backToFl.clearedLabels).toContain('References');
  });

  it('a checkpoint-to-checkpoint switch keeps a compatible component source', () => {
    const source = h3Model();
    const from = settingsFor(fl2vaTransformer(), { componentSourceModel: source, modelKey: fl2vaTransformer().key });
    const result = getVideoModelSelectionResult({
      currentSettings: from,
      model: ref2vaTransformer(),
      models: [source],
    });

    expect(result.settings.componentSourceModel?.key).toBe(source.key);
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

  it('the accelerator toggle resolves through the model-borne variant (4-step ref2v turbo)', () => {
    const model = ref2vaTransformer();
    const settings = settingsFor(model);
    const result = getAcceleratorToggleResult(settings, model, [model, REF2V_TURBO, FL2VA_TURBO], true);

    expect(result.missingLoras).toBe(false);
    expect(result.settings.acceleratorLoraKeys).toEqual(['ref2v-turbo']);
    expect(result.settings.steps).toBe(4);
  });

  const LIGHTX2V_REF2V_TURBO = {
    base: 'minimax-h3',
    key: 'lightx2v-ref2v-turbo',
    name: 'MiniMax H3 LightX2V Ref2V Turbo LoRA',
    type: 'lora' as const,
  };

  it('prefers the LightX2V ref2v release over the 4-step v0.1 repack whichever order they are listed in', () => {
    // Prefer the newer LightX2V release by version rather than alphabetical order.
    expect(findMiniMaxH3TurboLora([REF2V_TURBO, LIGHTX2V_REF2V_TURBO], { variant: 'ref2va' })).toMatchObject({
      key: 'lightx2v-ref2v-turbo',
    });
    expect(findMiniMaxH3TurboLora([LIGHTX2V_REF2V_TURBO, REF2V_TURBO], { variant: 'ref2va' })).toMatchObject({
      key: 'lightx2v-ref2v-turbo',
    });
    // Still a Ref2VA-only distillation: the FL2VA accelerator must not pick it up.
    expect(findMiniMaxH3TurboLora([LIGHTX2V_REF2V_TURBO, FL2VA_TURBO], { variant: 'fl2va' })).toMatchObject({
      key: 'turbo',
    });
  });

  it('a by-URL install of the LightX2V release, named by its file, still beats the old starter-named repack', () => {
    // Recognize URL-installed filename forms lacking the organization token so newer releases still outrank old
    // repacks.
    const rawNew = {
      base: 'minimax-h3',
      key: 'raw-new',
      name: 'minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16',
      type: 'lora' as const,
    };
    const rawOld = {
      base: 'minimax-h3',
      key: 'raw-old',
      name: 'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16',
      type: 'lora' as const,
    };

    expect(findMiniMaxH3TurboLora([REF2V_TURBO, rawNew], { variant: 'ref2va' })).toMatchObject({ key: 'raw-new' });
    expect(findMiniMaxH3TurboLora([rawNew, rawOld], { variant: 'ref2va' })).toMatchObject({ key: 'raw-new' });
    expect(findMiniMaxH3TurboLora([rawOld, rawNew], { variant: 'ref2va' })).toMatchObject({ key: 'raw-new' });
    // The raw file name states its schedule, so the toggle runs it at 8 steps.
    expect(getAcceleratorSteps(MINIMAX_H3_REF2V_TURBO_ACCELERATOR, [rawNew])).toBe(8);
    expect(getAcceleratorSteps(MINIMAX_H3_REF2V_TURBO_ACCELERATOR, [rawOld])).toBe(4);
  });

  it('a family-named older repack still beats a LightX2V-named look-alike without the family', () => {
    const lookAlike = { base: 'minimax-h3', key: 'look-alike', name: 'LightX2V Ref2V Turbo', type: 'lora' as const };

    expect(findMiniMaxH3TurboLora([lookAlike, REF2V_TURBO], { variant: 'ref2va' })).toMatchObject({
      key: 'ref2v-turbo',
    });
  });

  it('the accelerator toggle runs the LightX2V ref2v release at its 8-step schedule', () => {
    const model = ref2vaTransformer();
    const settings = settingsFor(model);
    const result = getAcceleratorToggleResult(
      settings,
      model,
      [model, REF2V_TURBO, LIGHTX2V_REF2V_TURBO, FL2VA_TURBO],
      true
    );

    expect(result.missingLoras).toBe(false);
    expect(result.settings.acceleratorLoraKeys).toEqual(['lightx2v-ref2v-turbo']);
    expect(result.settings.steps).toBe(8);
  });
});

describe('H3 component-source seeding', () => {
  it('defaults seed a component source for a checkpoint main, preferring a full install', () => {
    const checkpoint = h3Model('checkpoint');
    const componentsOnly = { ...h3Model('diffusers', 'h3-components'), components_only: true };
    const full = h3Model();

    expect(getDefaultVideoSettings(checkpoint, [componentsOnly, full]).componentSourceModel?.key).toBe(full.key);
    expect(getDefaultVideoSettings(checkpoint, [componentsOnly]).componentSourceModel?.key).toBe(componentsOnly.key);
    expect(getDefaultVideoSettings(checkpoint, []).componentSourceModel).toBeNull();
    // Diffusers mains never get one seeded.
    expect(getDefaultVideoSettings(full, [full]).componentSourceModel).toBeNull();
  });

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

    // FL2VA (and any non-reference model) still rejects the combination.
    const fl2va = h3Model();
    const reasons = getVideoValidationReasons(
      fl2va,
      settingsFor(fl2va, { references: [imageReference], sourceVideo: initialVideo })
    );

    expect(reasons).toContainEqual(expect.stringContaining('cannot be combined with an initial video on this model'));
  });

  it('a Wan -> Ref2VA switch snaps the frame count BEFORE sizing the tail window', () => {
    // Snap H3 generation frames before deriving its anchor; Wan's small count can produce unusably short context.
    for (const wanFrames of [5, 9, 21, 81]) {
      const wan = wanModel('i2v-14b');
      const toRef = getVideoModelSelectionResult({
        currentSettings: settingsFor(wan, { modelKey: wan.key, numFrames: wanFrames, sourceVideo: initialVideo }),
        model: ref2vaTransformer(),
        models: [wan],
      });
      const linked = toRef.settings.references.find(
        (entry) => entry.kind === 'video' && entry.fromSourceVideo === true
      );

      expect(toRef.settings.numFrames).toBe(90);
      // 90 frames of 24 fps material off a 24 fps clip, whatever Wan was on.
      expect(linked).toMatchObject({ clip: { endFrame: 400, startFrame: 311 } });
    }
  });

  it('an FL2VA -> Ref2VA switch keeps the initial video and derives its linked tail reference', () => {
    const fl2va = h3Model();
    const toRef = getVideoModelSelectionResult({
      currentSettings: settingsFor(fl2va, { modelKey: fl2va.key, sourceVideo: initialVideo }),
      model: ref2vaTransformer(),
      models: [fl2va],
    });

    expect(toRef.settings.sourceVideo).toEqual(initialVideo);
    expect(toRef.clearedLabels).not.toContain('Initial video');
    // Budget against the final snapped frame count: backend truncation removes overrun from the seam end.
    expect(toRef.settings.numFrames).toBe(124);
    expect(toRef.settings.references[0]).toMatchObject({
      clip: { endFrame: 400, startFrame: 277, video_name: 'long.mp4' },
      conditioning: 'video_audio',
      fromSourceVideo: true,
      kind: 'video',
    });

    // A task-neutral re-selection of the same model must NOT re-derive a
    // hand-tuned linked trim (only cutpoint changes do, via the setter).
    const tuned = {
      ...toRef.settings,
      references: toRef.settings.references.map((entry, index) =>
        index === 0 && entry.kind === 'video' ? { ...entry, clip: { ...entry.clip, startFrame: 300 } } : entry
      ),
    };
    const reselected = getVideoModelSelectionResult({
      currentSettings: tuned,
      model: ref2vaTransformer(),
      models: [fl2va],
    });

    expect(reselected.settings.references[0]).toMatchObject({ clip: { startFrame: 300 } });

    const backToFl = getVideoModelSelectionResult({
      currentSettings: toRef.settings,
      model: fl2va,
      models: [fl2va],
    });

    expect(backToFl.settings.references).toEqual([]);
    expect(backToFl.settings.sourceVideo).toEqual(initialVideo);
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
  it('offers every conditioning shape on both checkpoints', () => {
    const modes = ['txt2vid', 'first-frame', 'last-frame', 'first-last', 'extend', 'audio-to-video', 'video-to-audio'];

    expect(getVideoModes(ltx2('ltx2_dev'))).toEqual(modes);
    expect(getVideoModes(ltx2('ltx2_distilled'))).toEqual(modes);
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

      // A generated picture runs at whatever rate was asked for, so the soundtrack's four seconds
      // cover 120 frames there -- 113 after the snap -- and 89 at the clip's own 24.
      expect(audio).toEqual({ fps: 30, fpsFromClip: false, numFrames: 113, numFramesFromClip: true });
      expect(video).toEqual({ fps: 24, fpsFromClip: true, numFrames: 89, numFramesFromClip: true });
    });

    it("leaves the frame rate to the user when the gallery never knew the clip's own", () => {
      // A video record's fps is nullable, and `createVideoConditioningClip` falls back to 16 for
      // the length estimate. Playing and timing a held picture at that guess is the silent kind of
      // wrong: the clip comes out at the wrong speed with its soundtrack stretched to match.
      const guessed = getEffectiveVideoTiming(
        model,
        conditioned('video', { conditioningClip: { clip: CLIP, fpsKnown: false, role: 'video' }, fps: 30 })
      );

      expect(guessed).toMatchObject({ fps: 30, fpsFromClip: false });
      // The length still follows the clip: that comes from the frame count, which is known.
      expect(guessed.numFramesFromClip).toBe(true);
    });

    it("leaves the panel's own numbers alone when the model cannot run the mode", () => {
      // Stale state after a model switch: the panel offers to clear it rather than silently
      // deriving a length from a clip this family will never encode.
      const wan = wanModel('t2v_a14b');

      expect(
        getEffectiveVideoTiming(wan, {
          ...settingsFor(wan),
          conditioningClip: { clip: CLIP, fpsKnown: true, role: 'audio' },
        })
      ).toMatchObject({ numFramesFromClip: false });
    });

    it('is cleared by a model switch, like every other media slot', () => {
      // Left behind it still drives the canvas -- and it disables the aspect-ratio control while
      // doing so, so the panel could not be corrected from the panel.
      const wan = wanModel('t2v_a14b');
      const result = getVideoModelSelectionResult({ currentSettings: conditioned('video'), model: wan, models: [wan] });

      expect(result.settings.conditioningClip).toBeNull();
      expect(result.clearedLabels).toContain('Conditioning clip');
      expect(getVideoDimensions(wan, result.settings)?.source).toBe('aspect-ratio');
    });

    it('stops driving the canvas the moment the model cannot use it', () => {
      // The clip survives a reload on a family that cannot run it (a stored project, a recall).
      // Timing already falls back to the panel's own numbers in that case; the canvas must too.
      const wan = wanModel('t2v_a14b');
      const stale = { ...settingsFor(wan), conditioningClip: { clip: CLIP, fpsKnown: true, role: 'video' as const } };

      expect(getVideoDimensions(wan, stale)?.source).toBe('aspect-ratio');
    });

    it('refuses a continuation with no room left to continue', () => {
      // The generation opens by replaying the source's tail, so a frame count at or below that is
      // all context and no continuation. Left to the backend it fails after both encoders have run,
      // with a shape error about an anchor that does not fit its clip.
      const clip = {
        endFrame: 94,
        fps: 24,
        height: 704,
        numFrames: 96,
        startFrame: 0,
        video_name: 's.mp4',
        width: 1248,
      };
      const extending = (numFrames: number) =>
        settingsFor(model, {
          componentSourceModel: LTX2_COMPONENTS,
          ltx2TextEncoderModel: LTX2_ENCODER,
          numFrames,
          sourceVideo: clip,
        });

      expect(getVideoValidationReasons(model, extending(17)).join(' ')).toContain('Raise Frames above 17');
      expect(getVideoValidationReasons(model, extending(9)).join(' ')).toContain('Raise Frames above 17');
      expect(getVideoValidationReasons(model, extending(25))).toEqual([]);
    });

    it('refuses a source the join could not afford to blend', () => {
      // The crossfade is buffered at the SOURCE's own resolution, not the generation canvas, and
      // `video_concat` refuses over 512 MiB. Unchecked, that refusal lands after both encodes, the
      // transformer and the decode -- and neither remedy it names is reachable from the panel.
      const atSize = (width: number, height: number) =>
        settingsFor(model, {
          componentSourceModel: LTX2_COMPONENTS,
          ltx2TextEncoderModel: LTX2_ENCODER,
          numFrames: 121,
          sourceVideo: { endFrame: 94, fps: 24, height, numFrames: 96, startFrame: 0, video_name: 's.mp4', width },
        });

      expect(getVideoValidationReasons(model, atSize(2560, 1440))).toEqual([]);
      expect(getVideoValidationReasons(model, atSize(3840, 2160)).join(' ')).toContain('at or below about 2560x1440');
    });

    it('refuses a trim the join could not blend out of', () => {
      // The other side of the same arithmetic: the crossfade takes 17 frames from EACH half, so a
      // source trimmed shorter than that fails inside the join -- after the whole generation.
      const trimmed = (kept: number) =>
        settingsFor(model, {
          componentSourceModel: LTX2_COMPONENTS,
          ltx2TextEncoderModel: LTX2_ENCODER,
          numFrames: 121,
          sourceVideo: {
            endFrame: kept - 1,
            fps: 24,
            height: 704,
            numFrames: 96,
            startFrame: 0,
            video_name: 's.mp4',
            width: 1248,
          },
        });

      expect(getVideoValidationReasons(model, trimmed(10)).join(' ')).toContain('keeps only 10');
      expect(getVideoValidationReasons(model, trimmed(17))).toEqual([]);
    });

    it('derives the canvas from the clip only when its picture is the given one', () => {
      expect(getVideoDimensions(model, conditioned('video'))?.source).toBe('conditioning-clip');
      // In the audio role the picture is what gets generated, so the preset still owns the frame.
      expect(getVideoDimensions(model, conditioned('audio'))?.source).toBe('aspect-ratio');
    });

    it('refuses a clip beside any other conditioning slot', () => {
      expect(
        getVideoValidationReasons(
          model,
          conditioned('audio', { firstFrameImage: { height: 704, image_name: 'first.png', width: 1248 } })
        )
      ).toContain(
        'A conditioning clip cannot be combined with first/last frames, an initial video or references. Clear one side.'
      );
    });

    it('refuses a two-stage preset, which the denoise node cannot combine with a held modality', () => {
      expect(getVideoValidationReasons(model, conditioned('audio', { targetResolution: '1024p' })).join(' ')).toContain(
        'does not run a two-stage target resolution'
      );
      expect(getVideoValidationReasons(model, conditioned('audio'))).toEqual([]);
    });

    it('refuses a clip whose derived length falls outside the frame grid', () => {
      // Half a second of audio: under one frame group once snapped down.
      const short = { ...CLIP, numFrames: 8 };
      const long = { ...CLIP, numFrames: 24 * 60 };

      expect(
        getVideoValidationReasons(
          model,
          conditioned('audio', { conditioningClip: { clip: short, fpsKnown: true, role: 'audio' } })
        ).join(' ')
      ).toContain('Use a longer clip');
      expect(
        getVideoValidationReasons(
          model,
          conditioned('audio', { conditioningClip: { clip: long, fpsKnown: true, role: 'audio' } })
        ).join(' ')
      ).toContain('Use a shorter clip');
    });

    it("accepts a clip's fractional frame rate, which the panel's own field would reject", () => {
      // 29.97 is a real clip, and LTX-2's fps fields are floats.
      const ntsc = { ...CLIP, fps: 29.97 };

      expect(
        getVideoValidationReasons(
          model,
          conditioned('video', { conditioningClip: { clip: ntsc, fpsKnown: true, role: 'video' } })
        )
      ).toEqual([]);
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
    // The schedule is eight fixed noise levels; a step count the user could change would be a lie.
    expect(distilled.ui.stepsEditable).toBe(false);
    expect(distilled.defaults.steps).toBe(8);
  });

  it("seeds the release's own negative prompt, which dev guides against", () => {
    const settings = getDefaultVideoSettings(ltx2('ltx2_dev'), [LTX2_COMPONENTS, LTX2_ENCODER]);

    expect(settings.negativePrompt).toBe(LTX2_DEFAULT_NEGATIVE_PROMPT);
    expect(settings.negativePromptEnabled).toBe(true);
    // Families without one still start empty.
    expect(getDefaultVideoSettings(wanModel('t2v_a14b')).negativePrompt).toBe('');
  });

  it("restores dev's negative prompt when arriving from a variant that carries none", () => {
    // The distilled variant runs no unconditional pass and hides the field, so a panel used on it
    // reaches dev with nothing to guide against -- and dev would then encode an empty string at
    // CFG 3, which is the one part of the recipe silently missing.
    const catalog = [ltx2('ltx2_distilled'), ltx2('ltx2_dev'), LTX2_COMPONENTS, LTX2_ENCODER];
    const onDistilled = {
      ...getDefaultVideoSettings(ltx2('ltx2_distilled'), catalog),
      negativePrompt: '',
    };

    const toDev = getVideoModelSelectionResult({
      currentSettings: onDistilled,
      model: ltx2('ltx2_dev'),
      models: catalog,
    });

    expect(toDev.settings.negativePrompt).toBe(LTX2_DEFAULT_NEGATIVE_PROMPT);
    // Filling an empty field in is not something the panel took away.
    expect(toDev.clearedLabels).not.toContain('Negative prompt');
  });

  it('leaves an empty negative prompt alone when the panel came from a family that shows the field', () => {
    // Wan shows the field, so an empty box there is a value the user chose. Overwriting it would
    // also break recall, which relies on an empty recorded negative prompt leaving the panel's own
    // negative prompt untouched.
    const catalog = [wanModel('t2v_a14b'), ltx2('ltx2_dev'), LTX2_COMPONENTS, LTX2_ENCODER];
    const onWan = { ...getDefaultVideoSettings(wanModel('t2v_a14b'), catalog), negativePrompt: '' };

    const toDev = getVideoModelSelectionResult({ currentSettings: onWan, model: ltx2('ltx2_dev'), models: catalog });

    expect(toDev.settings.negativePrompt).toBe('');
  });

  it('keeps the policy and the canvas resolver agreeing on which presets are two-stage', () => {
    // Two places encode the stage count: the option's `stages` flag, which the panel and the graph
    // read, and `LTX2_TWO_STAGE_RESOLUTIONS`, which decides the 64 grid. A preset in one and not the
    // other snaps to the wrong grid or silently renders one pass -- with no error either way.
    for (const variant of ['ltx2_dev', 'ltx2_distilled']) {
      const model = ltx2(variant);
      const policy = getVideoModelPolicy(model, getDefaultVideoSettings(model));

      for (const option of policy.targetResolutions) {
        expect(
          isLtx2TwoStage(option.id as Ltx2TargetResolution),
          `${variant}/${option.id}: stages=${String(option.stages)}`
        ).toBe(option.stages === 2);
      }
    }
  });

  it('offers the two-stage presets and gives only dev a refine budget', () => {
    // The distilled schedule is fixed, so a refine budget would be ignored; dev pays four forwards
    // a step and would otherwise run its full 30 at four times the base pass's token count.
    const dev = getVideoModelPolicy(ltx2('ltx2_dev'), getDefaultVideoSettings(ltx2('ltx2_dev')));
    const distilled = getVideoModelPolicy(ltx2('ltx2_distilled'), getDefaultVideoSettings(ltx2('ltx2_distilled')));

    for (const policy of [dev, distilled]) {
      const twoStage = policy.targetResolutions.filter((option) => option.stages === 2);

      expect(twoStage.map((option) => option.id)).toEqual(['1024p', '1536p']);
    }
    expect(dev.refineSteps).toBe(8);
    expect(distilled.refineSteps).toBeUndefined();
    // The default stays a single pass: two-stage is a deliberate choice, not what a fresh panel does.
    expect(dev.targetResolutions.find((option) => option.id === dev.defaults.targetResolution)?.stages).toBeUndefined();
  });

  it("takes dev's steps and CFG from a panel that was never shown them", () => {
    // The distilled variant pins 8 steps at CFG 1 and hides both controls, and only the scales
    // refill on their own, so dev used to arrive at 8/1 -- dev with no guidance at all, which is
    // the recipe that produces washed-out output.
    const catalog = [ltx2('ltx2_distilled'), ltx2('ltx2_dev'), LTX2_COMPONENTS, LTX2_ENCODER];

    const result = getVideoModelSelectionResult({
      currentSettings: getDefaultVideoSettings(ltx2('ltx2_distilled'), catalog),
      model: ltx2('ltx2_dev'),
      models: catalog,
    });

    expect({ cfgScale: result.settings.cfgScale, steps: result.settings.steps }).toEqual({ cfgScale: 3, steps: 30 });
  });

  it("leaves another family's own numbers alone when its panel was showing them", () => {
    // The deliberate boundary: Wan shows steps and CFG, so 40/5 on screen is the user's to change
    // and LTX-2 dev does not overrule it. Only a control the previous variant never showed is
    // treated as a recommendation the panel was merely holding.
    const catalog = [wanModel('t2v_a14b'), ltx2('ltx2_dev'), LTX2_COMPONENTS, LTX2_ENCODER];
    const onWan = getDefaultVideoSettings(wanModel('t2v_a14b'), catalog);

    const toDev = getVideoModelSelectionResult({ currentSettings: onWan, model: ltx2('ltx2_dev'), models: catalog });

    expect({ cfgScale: toDev.settings.cfgScale, steps: toDev.settings.steps }).toEqual({
      cfgScale: onWan.cfgScale,
      steps: onWan.steps,
    });
    expect(onWan.steps).not.toBe(30);
  });

  it('never rewrites steps or CFG the previous panel put on screen, whatever they hold', () => {
    // Wan shows both controls, so its numbers are the user's whatever they are -- including numbers
    // that happen to match a default. The comparison is against the variant's static recipe for
    // this reason: getDefaultVideoSettings applies an installed accelerator, so with the Lightning
    // pair in the catalog Wan's "default" is the 4-step fast path, and a hand-typed 4 at CFG 1
    // would read as a carried-over default and be rewritten to 40/5.
    const catalog = [wanModel('t2v_a14b'), ltx2('ltx2_dev'), LIGHTNING_T2V_HIGH, LIGHTNING_T2V_LOW, LTX2_COMPONENTS];
    const typed = {
      ...getDefaultVideoSettings(wanModel('t2v_a14b'), catalog),
      acceleratorEnabled: false,
      acceleratorLoraKeys: [],
      cfgScale: 1,
      loras: [],
      steps: 4,
    };

    const toDev = getVideoModelSelectionResult({ currentSettings: typed, model: ltx2('ltx2_dev'), models: catalog });

    expect({ cfgScale: toDev.settings.cfgScale, steps: toDev.settings.steps }).toEqual({ cfgScale: 1, steps: 4 });
  });

  it('keeps a tuned CFG across a detour through the fixed-schedule variant', () => {
    // The mirror of the rule above: a number the user chose is not the previous model's
    // recommendation, so it survives a variant that hides the control. The step count cannot --
    // distilled pins it on the way in and says so -- but it must land on dev's 30, not distilled's 8.
    const catalog = [ltx2('ltx2_distilled'), ltx2('ltx2_dev'), LTX2_COMPONENTS, LTX2_ENCODER];
    const tuned = { ...getDefaultVideoSettings(ltx2('ltx2_dev'), catalog), cfgScale: 6, steps: 45 };

    const toDistilled = getVideoModelSelectionResult({
      currentSettings: tuned,
      model: ltx2('ltx2_distilled'),
      models: catalog,
    });

    expect(toDistilled.settings.cfgScale).toBe(6);
    expect(toDistilled.clearedLabels).toContain('Steps');

    const back = getVideoModelSelectionResult({
      currentSettings: toDistilled.settings,
      model: ltx2('ltx2_dev'),
      models: catalog,
    });

    expect({ cfgScale: back.settings.cfgScale, steps: back.settings.steps }).toEqual({ cfgScale: 6, steps: 30 });
  });

  it('leaves steps and CFG alone when the panel it came from cannot be resolved', () => {
    // Without the previous model there is nothing to compare against, and guessing would clobber
    // tuned values on any panel whose model is missing from the catalog.
    const catalog = [ltx2('ltx2_dev'), LTX2_COMPONENTS, LTX2_ENCODER];
    const orphaned = {
      ...getDefaultVideoSettings(ltx2('ltx2_distilled'), catalog),
      cfgScale: 6,
      modelKey: 'uninstalled-key',
      steps: 45,
    };

    const toDev = getVideoModelSelectionResult({ currentSettings: orphaned, model: ltx2('ltx2_dev'), models: catalog });

    expect({ cfgScale: toDev.settings.cfgScale, steps: toDev.settings.steps }).toEqual({ cfgScale: 6, steps: 45 });
  });

  it('seeds the list when the panel it came from can no longer be resolved', () => {
    // The previous model was uninstalled under the panel, or the panel was never seeded at all. The
    // automatic re-pick has to reach the same place the manual switch does, or dev runs at CFG 3
    // against an empty string depending on how it was selected.
    const catalog = [ltx2('ltx2_dev'), LTX2_COMPONENTS, LTX2_ENCODER];
    const orphaned = {
      ...getDefaultVideoSettings(ltx2('ltx2_distilled'), catalog),
      modelKey: 'uninstalled-key',
      negativePrompt: '',
    };

    const toDev = getVideoModelSelectionResult({ currentSettings: orphaned, model: ltx2('ltx2_dev'), models: catalog });

    expect(toDev.settings.negativePrompt).toBe(LTX2_DEFAULT_NEGATIVE_PROMPT);
  });

  it('does not put a list into a negative prompt field the user switched off', () => {
    const catalog = [ltx2('ltx2_distilled'), ltx2('ltx2_dev'), LTX2_COMPONENTS, LTX2_ENCODER];
    const disabled = {
      ...getDefaultVideoSettings(ltx2('ltx2_distilled'), catalog),
      negativePrompt: '',
      negativePromptEnabled: false,
    };

    const toDev = getVideoModelSelectionResult({ currentSettings: disabled, model: ltx2('ltx2_dev'), models: catalog });

    expect(toDev.settings.negativePrompt).toBe('');
    expect(toDev.settings.negativePromptEnabled).toBe(false);
  });

  it('carries a written negative prompt through a detour rather than overwriting it', () => {
    const catalog = [ltx2('ltx2_distilled'), ltx2('ltx2_dev'), LTX2_COMPONENTS, LTX2_ENCODER];
    const written = 'shaky handheld footage, lens flare';
    const onDev = {
      ...getDefaultVideoSettings(ltx2('ltx2_dev'), catalog),
      negativePrompt: written,
    };

    const toDistilled = getVideoModelSelectionResult({
      currentSettings: onDev,
      model: ltx2('ltx2_distilled'),
      models: catalog,
    });
    const backToDev = getVideoModelSelectionResult({
      currentSettings: toDistilled.settings,
      model: ltx2('ltx2_dev'),
      models: catalog,
    });

    expect(backToDev.settings.negativePrompt).toBe(written);
  });

  it('keeps a hidden CFG so a detour through another family does not destroy it', () => {
    // MiniMax H3 hides CFG and never reads it; a value the user tuned for Wan has to survive
    // selecting H3 and coming back.
    const wan = wanModel('t2v_a14b');
    const tuned = { ...getDefaultVideoSettings(wan), cfgScale: 8 };
    const viaH3 = getVideoModelSelectionResult({ currentSettings: tuned, model: h3Model(), models: [] });
    const back = getVideoModelSelectionResult({ currentSettings: viaH3.settings, model: wan, models: [] });

    expect(back.settings.cfgScale).toBe(8);
  });

  it('seeds the components and encoder a picked model cannot run without', () => {
    const model = ltx2('ltx2_dev');
    const settings = getDefaultVideoSettings(model, [LTX2_COMPONENTS, LTX2_ENCODER]);

    expect(settings.componentSourceModel).toEqual(LTX2_COMPONENTS);
    expect(settings.ltx2TextEncoderModel).toEqual(LTX2_ENCODER);
    expect(settings.audioCfgScale).toBe(7);
    expect(settings.stgScale).toBe(1);
    expect(settings.modalityScale).toBe(3);
  });

  it('requires the Gemma-4 encoder on every LTX-2 model shape', () => {
    for (const model of [ltx2('ltx2_dev'), ltx2('ltx2_dev', { format: 'diffusers', key: 'ltx2-folder' })]) {
      const settings = { ...getDefaultVideoSettings(model), ltx2TextEncoderModel: null };

      expect(getVideoValidationReasons(model, { ...settings, componentSourceModel: LTX2_COMPONENTS })).toContain(
        'LTX-2 needs its Gemma-4 text encoder — no LTX-2 model carries one.'
      );
    }
  });

  it('names the fix for a components-only install picked as the model', () => {
    const componentsOnly = ltx2('ltx2_dev', {
      components_only: true,
      format: 'diffusers',
      key: 'ltx2-components-only',
      name: 'LTX-2.5 Components',
    } as Partial<MainModelConfig>);

    expect(isVideoModelSelectable(componentsOnly as GenerationModelCatalogItem)).toBe(false);
    expect(getVideoValidationReasons(componentsOnly, getDefaultVideoSettings(componentsOnly))[0]).toContain(
      'single-file LTX-2 transformer'
    );
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

  it('derives the canvas from the short-edge preset', () => {
    const model = ltx2('ltx2_dev');
    const settings = getDefaultVideoSettings(model);

    expect(getVideoDimensions(model, settings)).toEqual({ height: 704, source: 'aspect-ratio', width: 1248 });
    expect(getVideoDimensions(model, { ...settings, targetResolution: '768p' })).toEqual({
      height: 768,
      source: 'aspect-ratio',
      width: 1376,
    });
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
    // Filling in controls the panel did not have a moment ago is not a clearing, and reporting it
    // would name three settings the user has never seen.
    expect(toDev.clearedLabels).not.toContain('Advanced guidance');

    // Moving to the fixed schedule drops both the scales and a carried-over step count.
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
    expect(toDistilled.clearedLabels).toEqual(expect.arrayContaining(['Steps', 'Advanced guidance']));
  });

  it('keeps the negative prompt whenever either classifier-free scale consumes it', () => {
    // One unconditional pass serves both streams (LTX2Guidance.passes), so the panel's rule has to
    // be the same OR the backend's — a mismatch queues a graph the denoise node refuses.
    const model = ltx2('ltx2_dev');
    const base = getDefaultVideoSettings(model, [LTX2_COMPONENTS, LTX2_ENCODER]);
    const used = (overrides: Partial<VideoSettings>) =>
      getVideoPromptPolicy(model, { ...base, ...overrides }).negativeUsedInGraph;

    expect(used({ cfgScale: 3, audioCfgScale: 7 })).toBe(true);
    expect(used({ cfgScale: 1, audioCfgScale: 7 })).toBe(true);
    expect(used({ cfgScale: 3, audioCfgScale: 1 })).toBe(true);
    expect(used({ cfgScale: 1, audioCfgScale: 1 })).toBe(false);
    expect(used({ cfgScale: 3, audioCfgScale: 7, negativePromptEnabled: false })).toBe(false);
  });

  it('accepts its own target-resolution presets as stored values', () => {
    // The panel writes the preset through `isVideoTargetResolution`; a preset the guard does not
    // know is dropped on the way in, leaving the control inert and healing the stored value to
    // another family's preset.
    const model = ltx2('ltx2_dev');

    for (const option of getVideoModelPolicy(model, getDefaultVideoSettings(model)).targetResolutions) {
      expect(isVideoTargetResolution(option.id), `${option.id} is not accepted`).toBe(true);

      const healed = normalizeVideoSettings({
        ...getDefaultVideoSettings(model),
        targetResolution: option.id,
      });

      expect(healed?.targetResolution).toBe(option.id);
    }
  });

  it('rejects a guidance scale the denoise node would refuse at enqueue', () => {
    const model = ltx2('ltx2_dev');
    const settings = {
      ...getDefaultVideoSettings(model, [LTX2_COMPONENTS, LTX2_ENCODER]),
      audioCfgScale: 0.5,
      modalityScale: 0.2,
      stgScale: -1,
    };
    const reasons = getVideoValidationReasons(model, settings);

    expect(reasons).toContain('Audio CFG must be at least 1.');
    expect(reasons).toContain('STG must be at least 0.');
    expect(reasons).toContain('Modality guidance must be at least 1.');
  });
});
