import type { ComponentModelConfig, GenerateModelConfig, VaeModelConfig } from '@features/generation/contracts';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import {
  getDefaultGenerateSettings,
  getMaxReferenceImages,
  isSupportedGenerateModel,
  isVaeModelConfig,
} from '@features/generation/settings';
import { describe, expect, it } from 'vitest';

import { getCurrentGenerateValues } from './executeImageRecall';
import {
  buildRecallParametersSettings,
  getRecallParametersSkipMessage,
  isRecallParametersUpdatedEvent,
} from './recallParameters';

const makeModel = (key: string, base: string, type: string, extra: Record<string, unknown> = {}) =>
  ({ base, hash: `${key}-hash`, key, name: `${key} name`, type, ...extra }) as ComponentModelConfig;

const sdxl = makeModel('sdxl-main', 'sdxl', 'main');
const sd1 = makeModel('sd1-main', 'sd-1', 'main');
const flux = makeModel('flux-main', 'flux', 'main');
const sd3 = makeModel('sd3-main', 'sd-3', 'main');
const sdxlLora = makeModel('sdxl-lora', 'sdxl', 'lora');
const fluxLora = makeModel('flux-lora', 'flux', 'lora');
const sdxlIpAdapter = makeModel('sdxl-ipa', 'sdxl', 'ip_adapter');
const fluxRedux = makeModel('flux-redux', 'flux', 'flux_redux');
const sdxlVae = makeModel('sdxl-vae', 'sdxl', 'vae');
const models = [sdxl, sd1, flux, sd3, sdxlLora, fluxLora, sdxlIpAdapter, fluxRedux, sdxlVae];
const supportedModels = models.filter(isSupportedGenerateModel) as GenerateModelConfig[];
const vaeModels = models.filter(isVaeModelConfig) as VaeModelConfig[];

const image = (imageName: string) => ({ height: 512, image_name: imageName, width: 512 });

const currentFor = (model: ComponentModelConfig, overrides: Record<string, unknown> = {}) => {
  const values = getCurrentGenerateValues({ generateValues: { modelKey: model.key }, supportedModels });
  if (!values) {
    throw new Error('fixture model is not a supported Generate model');
  }
  return { ...values, ...overrides };
};

const build = (currentValues: ReturnType<typeof currentFor>, parameters: Record<string, unknown>) =>
  buildRecallParametersSettings({ currentValues, models, parameters, supportedModels, vaeModels });

seedArchitectureCapabilities();

describe('buildRecallParametersSettings', () => {
  it('applies prompts, seed, size and sampler parameters to the current values', () => {
    const result = build(currentFor(sdxl), {
      cfg_scale: 7.5,
      height: 768,
      negative_prompt: 'blurry',
      positive_prompt: 'a cat',
      scheduler: 'euler',
      seamless_x: true,
      seed: 42,
      steps: 30,
      width: 1024,
    });

    expect(result.skipped).toEqual([]);
    expect(result.fields).toEqual(
      expect.arrayContaining(['prompts', 'seed', 'steps', 'cfg', 'scheduler', 'seamless', 'size'])
    );
    expect(result.values).toEqual(
      expect.objectContaining({
        aspectRatioId: '4:3',
        cfgScale: 7.5,
        height: 768,
        negativePrompt: 'blurry',
        positivePrompt: 'a cat',
        scheduler: 'euler',
        seamlessXAxis: true,
        seed: 42,
        seedMode: 'fixed',
        steps: 30,
        width: 1024,
      })
    );
  });

  it('applies CLIP skip only where the model family exposes it', () => {
    expect(build(currentFor(sd1), { clip_skip: 2 }).values.clipSkip).toBe(2);

    const strictReset = build(currentFor(sdxl, { clipSkip: 3 }), { clip_skip: null });
    expect(strictReset.values.clipSkip).toBe(0);
    expect(strictReset.skipped).toEqual([]);
    expect(build(currentFor(sdxl), { clip_skip: 2 }).skipped).toEqual([
      { detail: sdxl.name, key: 'clip_skip', reason: 'incompatible' },
    ]);
  });

  it('switches model with its defaults, then lets explicit parameters override them', () => {
    const result = build(currentFor(sdxl, { steps: 50 }), { model: flux.key, steps: 12 });

    expect(result.fields).toEqual(expect.arrayContaining(['model', 'steps']));
    expect(result.values.model.key).toBe(flux.key);
    expect(result.values.modelKey).toBe(flux.key);
    expect(result.values.steps).toBe(12);
    expect(result.values.cfgScale).toBe(getDefaultGenerateSettings(flux as GenerateModelConfig).cfgScale);
  });

  it('retargets reference images and clears an incompatible VAE when the model family changes', () => {
    const current = currentFor(sdxl, {
      referenceImages: [
        {
          config: { image: { original: { image: image('kept.png') } }, model: sdxlIpAdapter, type: 'ip_adapter' },
          id: 'kept',
          isEnabled: true,
        },
      ],
      vae: sdxlVae,
    });
    const result = build(current, { model: flux.key });

    expect(result.fields).toEqual(['model']);
    expect(result.values.vae).toBeNull();
    expect(result.values.referenceImages).toHaveLength(1);
    expect(result.values.referenceImages[0]?.config).toEqual(
      expect.objectContaining({
        clipVisionModel: 'ViT-L',
        image: { original: { image: image('kept.png') } },
        model: null,
        type: 'ip_adapter',
      })
    );
  });

  it('resets the VAE to the model default on a null (strict-mode) vae_model', () => {
    const result = build(currentFor(sdxl, { vae: sdxlVae }), { vae_model: null });

    expect(result.fields).toEqual(['vae']);
    expect(result.values.vae).toBeNull();
  });

  it('reports an unknown model and a VAE from another family without changing the selection', () => {
    const current = currentFor(sdxl);
    const result = build(current, { model: 'not-installed', vae_model: sdxlVae.name });

    expect(result.values.model.key).toBe(sdxl.key);
    expect(result.values.vae?.key).toBe(sdxlVae.key);
    expect(result.skipped).toEqual([{ detail: 'not-installed', key: 'model', reason: 'unresolved' }]);

    const fluxResult = build(currentFor(flux), { vae_model: sdxlVae.key });
    expect(fluxResult.values.vae).toBeNull();
    expect(fluxResult.skipped).toEqual([{ detail: sdxlVae.name, key: 'vae_model', reason: 'incompatible' }]);
  });

  it('resets null (strict-mode) parameters to the model defaults and clears empty lists', () => {
    const existingReference = {
      config: { image: { original: { image: image('old.png') } }, type: 'ip_adapter' as const },
      id: 'existing',
      isEnabled: true,
    };
    const current = currentFor(sdxl, {
      loras: [{ isEnabled: true, model: sdxlLora, weight: 1 }],
      positivePrompt: 'keep me?',
      referenceImages: [{ ...existingReference, config: { ...existingReference.config, model: sdxlIpAdapter } }],
      seed: 7,
      seedMode: 'fixed',
      steps: 50,
      width: 1536,
    });
    const result = build(current, {
      control_layers: [],
      ip_adapters: [],
      loras: [],
      model: null,
      positive_prompt: null,
      reference_images: [],
      refiner_model: null,
      seed: null,
      steps: null,
      width: null,
    });
    const defaults = getDefaultGenerateSettings(sdxl as GenerateModelConfig);

    expect(result.skipped).toEqual([]);
    expect(result.values).toEqual(
      expect.objectContaining({
        loras: [],
        positivePrompt: '',
        referenceImages: [],
        seedMode: 'random',
        steps: defaults.steps,
        width: defaults.width,
      })
    );
    expect(result.values.model.key).toBe(sdxl.key);
  });

  it('builds the LoRA list from installed models, dropping unknown, incompatible and duplicate entries', () => {
    const result = build(currentFor(sdxl), {
      loras: [
        { is_enabled: false, model_key: sdxlLora.key, weight: 0.5 },
        { model_key: fluxLora.key },
        { model_key: 'missing-lora' },
        { model_key: sdxlLora.key, weight: 20 },
      ],
    });

    expect(result.fields).toEqual(['loras']);
    expect(result.values.loras).toEqual([{ isEnabled: false, model: sdxlLora, weight: 0.5 }]);
    expect(build(currentFor(sdxl), { loras: [{ model_key: sdxlLora.key }] }).values.loras).toEqual([
      { isEnabled: true, model: sdxlLora, weight: 0.75 },
    ]);
    expect(result.skipped).toEqual([
      { detail: fluxLora.name, key: 'loras', reason: 'incompatible' },
      { detail: 'missing-lora', key: 'loras', reason: 'unresolved' },
    ]);
  });

  it('builds IP adapter, FLUX Redux and model-free reference images for the effective model', () => {
    const result = build(currentFor(sdxl), {
      ip_adapters: [
        {
          begin_step_percent: 0.1,
          image: image('style.png'),
          method: 'style',
          model_key: sdxlIpAdapter.key,
          weight: 0.7,
        },
        { model_key: 'missing-adapter' },
        { model_key: sdxlIpAdapter.key },
      ],
      reference_images: [{ image: image('subject.png') }, { image: { image_name: '' } }],
    });

    expect(result.fields).toEqual(['referenceImages']);
    expect(result.values.referenceImages).toHaveLength(2);
    expect(result.values.referenceImages[0]?.config).toEqual({
      beginEndStepPct: [0.1, 1],
      clipVisionModel: 'ViT-H',
      image: { original: { image: image('style.png') } },
      method: 'style',
      model: sdxlIpAdapter,
      type: 'ip_adapter',
      weight: 0.7,
    });
    expect(result.values.referenceImages[1]?.config).toEqual(
      expect.objectContaining({ image: { original: { image: image('subject.png') } }, type: 'ip_adapter' })
    );
    expect(result.skipped).toEqual([
      { detail: 'missing-adapter', key: 'ip_adapters', reason: 'unresolved' },
      { detail: `${sdxlIpAdapter.key}: image missing`, key: 'ip_adapters', reason: 'unresolved' },
      { key: 'reference_images', reason: 'invalid' },
    ]);

    const redux = build(currentFor(flux), {
      ip_adapters: [{ image: image('redux.png'), image_influence: 'low', model_key: fluxRedux.key }],
    });
    expect(redux.values.referenceImages[0]?.config).toEqual({
      image: { original: { image: image('redux.png') } },
      imageInfluence: 'low',
      model: fluxRedux,
      type: 'flux_redux',
    });
  });

  it('appends reference images behind the existing list and enforces the model limit', () => {
    const current = currentFor(sdxl, {
      referenceImages: [
        {
          config: { image: { original: { image: image('existing.png') } }, type: 'ip_adapter', model: sdxlIpAdapter },
          id: 'existing',
          isEnabled: true,
        },
      ],
    });
    const appended = build(current, { append: true, reference_images: [{ image: image('new.png') }] });

    expect(
      appended.values.referenceImages.map((reference) => reference.config.image?.original.image.image_name)
    ).toEqual(['existing.png', 'new.png']);

    const limit = getMaxReferenceImages(sdxl as GenerateModelConfig);
    const overLimit = build(current, {
      append: true,
      reference_images: Array.from({ length: limit }, (_, index) => ({ image: image(`extra-${index}.png`) })),
    });

    expect(overLimit.values.referenceImages).toHaveLength(limit);
    expect(overLimit.skipped).toEqual([{ detail: String(limit), key: 'reference_images', reason: 'limit' }]);

    const untouched = build(current, { append: true, reference_images: [] });
    expect(untouched.fields).toEqual([]);
    expect(untouched.values.referenceImages).toEqual(current.referenceImages);
  });

  it('drops reference images for a model without reference image support and says so', () => {
    const result = build(currentFor(sd3), { reference_images: [{ image: image('subject.png') }] });

    expect(result.values.referenceImages).toEqual([]);
    expect(result.skipped).toEqual([{ detail: sd3.name, key: 'reference_images', reason: 'incompatible' }]);
  });

  it('reports parameters the Generate panel has no home for, ignoring empty lists', () => {
    const result = build(currentFor(sdxl), {
      control_layers: [{ model_key: 'controlnet' }],
      denoise_strength: 0.5,
      positive_prompt: 'still applied',
      refiner_model: 'refiner',
    });

    expect(result.fields).toEqual(['prompts']);
    expect(result.skipped.map((skip) => skip.key).sort()).toEqual([
      'control_layers',
      'denoise_strength',
      'refiner_model',
    ]);
    expect(result.skipped.every((skip) => skip.reason === 'unsupported')).toBe(true);

    expect(build(currentFor(sdxl), { control_layers: [] }).skipped).toEqual([]);
  });

  it('maps guidance onto cfgScale only for guidance-driven model families', () => {
    expect(build(currentFor(flux), { guidance: 3.5 }).values.cfgScale).toBe(3.5);
    expect(build(currentFor(sdxl), { guidance: 3.5 }).skipped).toEqual([
      { detail: sdxl.name, key: 'guidance', reason: 'incompatible' },
    ]);

    const both = build(currentFor(flux), { cfg_scale: 4, guidance: 3.5 });
    expect(both.values.cfgScale).toBe(4);
    expect(both.skipped).toEqual([{ detail: 'cfg_scale was also provided', key: 'guidance', reason: 'unsupported' }]);

    const invalidCfg = build(currentFor(flux), { cfg_scale: 0.5, guidance: 3.5 });
    expect(invalidCfg.values.cfgScale).toBe(3.5);
    expect(invalidCfg.skipped).toEqual([{ key: 'cfg_scale', reason: 'invalid' }]);
  });

  it('flags out-of-range scalars instead of applying them', () => {
    const result = build(currentFor(sdxl), { seed: -1, steps: 0, width: 10 });

    expect(result.fields).toEqual([]);
    expect(result.skipped.map((skip) => skip.key).sort()).toEqual(['seed', 'steps', 'width']);
  });
});

describe('getRecallParametersSkipMessage', () => {
  it('names every skipped parameter with its reason and detail', () => {
    expect(
      getRecallParametersSkipMessage([
        { detail: 'pixel-art', key: 'loras', reason: 'incompatible' },
        { key: 'refiner_model', reason: 'unsupported' },
        { detail: '5', key: 'reference_images', reason: 'limit' },
      ])
    ).toBe(
      'loras: incompatible with the current model (pixel-art); refiner_model: not supported by the Generate panel; reference_images: over the reference image limit (5)'
    );
  });
});

describe('isRecallParametersUpdatedEvent', () => {
  it('accepts the backend envelope and rejects payloads without a parameters object', () => {
    expect(isRecallParametersUpdatedEvent({ parameters: { steps: 1 }, queue_id: 'default', user_id: 'u' })).toBe(true);
    expect(isRecallParametersUpdatedEvent({ parameters: [], queue_id: 'default', user_id: 'u' })).toBe(false);
    expect(isRecallParametersUpdatedEvent({ queue_id: 'default', user_id: 'u' })).toBe(false);
    expect(isRecallParametersUpdatedEvent(null)).toBe(false);
  });
});
