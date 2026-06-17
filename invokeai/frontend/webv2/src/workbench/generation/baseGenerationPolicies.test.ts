import { describe, expect, it } from 'vitest';

import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
  MainModelConfig,
  VaeModelConfig,
} from './types';

import {
  BASE_GENERATION,
  coerceSchedulerForGraph,
  getComponentSectionPolicy,
  getAutoFlux2ComponentSourceModel,
  getDefaultGenerateSettings,
  getGenerationDimensions,
  getGenerationModelAvailabilityReasons,
  getGenerationModelPolicy,
  getGenerationValidationReasons,
  getPromptPolicy,
  getSettingsWithCompatibleModelSelections,
  isSupportedGenerateModel,
  SUPPORTED_GENERATE_BASES,
} from './baseGenerationPolicies';

const createModel = (base: string, overrides: Partial<MainModelConfig> = {}): MainModelConfig => ({
  base,
  key: `${base}-model`,
  name: `${base} model`,
  type: 'main',
  ...overrides,
});

const createSettings = (model: GenerateModelConfig, overrides: Partial<GenerateSettings> = {}): GenerateSettings => ({
  ...getDefaultGenerateSettings(model),
  seed: 1,
  shouldRandomizeSeed: false,
  ...overrides,
});

const t5Encoder: ComponentModelConfig = { base: 'any', key: 't5', name: 'T5 Encoder', type: 't5_encoder' };
const clipEmbed: ComponentModelConfig = { base: 'any', key: 'clip', name: 'CLIP Embed', type: 'clip_embed' };
const qwenVlEncoder: ComponentModelConfig = {
  base: 'any',
  key: 'qwen-vl',
  name: 'Qwen VL Encoder',
  type: 'qwen_vl_encoder',
};
const qwen3Encoder: ComponentModelConfig = {
  base: 'any',
  key: 'qwen3',
  name: 'Qwen3 Encoder',
  type: 'qwen3_encoder',
  variant: 'qwen3_8b',
};
const animaQwen3Encoder: ComponentModelConfig = {
  base: 'any',
  key: 'anima-qwen3',
  name: 'Anima Qwen3 Encoder',
  type: 'qwen3_encoder',
  variant: 'qwen3_06b',
};
const fluxVae: VaeModelConfig = { base: 'flux', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' };
const flux2Vae: VaeModelConfig = { base: 'flux2', key: 'flux2-vae', name: 'FLUX.2 VAE', type: 'vae' };
const qwenImageVae: VaeModelConfig = { base: 'qwen-image', key: 'qwen-vae', name: 'Qwen VAE', type: 'vae' };
const sdxlVae: VaeModelConfig = { base: 'sdxl', key: 'sdxl-vae', name: 'SDXL VAE', type: 'vae' };
const sd1Lora: LoraModelConfig = { base: 'sd-1', key: 'sd1-lora', name: 'SD 1 LoRA', type: 'lora' };
const sdxlLora: LoraModelConfig = { base: 'sdxl', key: 'sdxl-lora', name: 'SDXL LoRA', type: 'lora' };
const externalModel: GenerateModelConfig = {
  base: 'external',
  capabilities: { modes: ['txt2img'], supports_negative_prompt: false, supports_seed: false },
  format: 'external_api',
  key: 'external-model',
  name: 'External Model',
  provider_id: 'openai',
  type: 'external_image_generator',
};

describe('BASE_GENERATION', () => {
  it('matches expected dimensions per base', () => {
    expect(getGenerationDimensions(createModel('sd-1'))).toMatchObject({ grid: 8, optimal: 512 });
    expect(getGenerationDimensions(createModel('sdxl'))).toMatchObject({ grid: 8, optimal: 1024 });
    expect(getGenerationDimensions(createModel('flux2'))).toMatchObject({ grid: 16, optimal: 1024 });
    expect(getGenerationDimensions(createModel('cogview4'))).toMatchObject({ grid: 32, optimal: 1024 });
  });

  it('matches expected defaults per base', () => {
    expect(getDefaultGenerateSettings(createModel('sdxl'))).toMatchObject({
      steps: 30,
      cfgScale: 7,
      scheduler: 'euler_a',
      width: 1024,
      height: 1024,
    });
    expect(getDefaultGenerateSettings(createModel('flux'))).toMatchObject({
      steps: 4,
      cfgScale: 4,
      scheduler: 'euler',
    });
    expect(getDefaultGenerateSettings(createModel('flux2'))).toMatchObject({
      steps: 4,
      cfgScale: 1,
      scheduler: 'euler',
    });
    expect(getDefaultGenerateSettings(createModel('qwen-image'))).toMatchObject({
      steps: 40,
      cfgScale: 4,
      scheduler: 'euler_a',
    });
    expect(getDefaultGenerateSettings(createModel('z-image'))).toMatchObject({
      steps: 8,
      cfgScale: 1,
      scheduler: 'euler',
    });
  });

  it('matches expected scheduler sets per base', () => {
    const flux = createModel('flux');
    const zbase = createModel('z-image', { variant: 'zbase' });
    const anima = createModel('anima');

    expect(
      getGenerationModelPolicy(createModel('sdxl'), createSettings(createModel('sdxl'))).scheduler.options.map(
        (option) => option.value
      )
    ).toContain('euler_a');
    expect(
      getGenerationModelPolicy(flux, createSettings(flux)).scheduler.options.map((option) => option.value)
    ).toEqual(['euler', 'heun', 'lcm']);
    expect(
      getGenerationModelPolicy(zbase, createSettings(zbase)).scheduler.options.map((option) => option.value)
    ).toEqual(['euler', 'heun']);
    expect(
      getGenerationModelPolicy(anima, createSettings(anima)).scheduler.options.map((option) => option.value)
    ).toEqual(['euler', 'heun', 'dpmpp_2m', 'dpmpp_2m_sde', 'er_sde', 'lcm']);
    expect(coerceSchedulerForGraph(zbase, 'lcm')).toBe('euler');
  });

  it('matches expected prompt policy per base', () => {
    expect(getPromptPolicy(createModel('flux'), { cfgScale: 4 })).toMatchObject({
      negativeVisible: false,
      negativeUsedInGraph: false,
    });
    expect(getPromptPolicy(createModel('sdxl'), { cfgScale: 1 })).toMatchObject({
      negativeVisible: true,
      negativeUsedInGraph: true,
    });
    expect(getPromptPolicy(createModel('qwen-image'), { cfgScale: 1 })).toMatchObject({
      negativeVisible: true,
      negativeUsedInGraph: false,
    });
    expect(getPromptPolicy(createModel('qwen-image'), { cfgScale: 2 })).toMatchObject({
      negativeVisible: true,
      negativeUsedInGraph: true,
      negativeHelpText: 'Used only when CFG is greater than 1.',
    });
    expect(getPromptPolicy(externalModel, { cfgScale: 7 })).toMatchObject({
      negativeVisible: false,
      negativeUsedInGraph: false,
    });
    expect(
      getPromptPolicy({ ...externalModel, capabilities: { supports_negative_prompt: true } }, { cfgScale: 7 })
    ).toMatchObject({ negativeVisible: true, negativeUsedInGraph: true });
  });

  it('matches expected UI availability per base', () => {
    expect(getGenerationModelPolicy(createModel('sd-1'), createSettings(createModel('sd-1'))).ui).toMatchObject({
      clipSkipMax: 12,
      cfgRescaleVisible: true,
      schedulerVisible: true,
    });
    expect(getGenerationModelPolicy(createModel('sd-2'), createSettings(createModel('sd-2'))).ui).toMatchObject({
      clipSkipMax: 24,
      cfgRescaleVisible: true,
    });
    expect(getGenerationModelPolicy(createModel('flux'), createSettings(createModel('flux'))).ui).toMatchObject({
      guidanceLabel: 'Guidance',
      schedulerVisible: true,
      clipSkipMax: null,
    });
    expect(getGenerationModelPolicy(createModel('sd-3'), createSettings(createModel('sd-3'))).ui).toMatchObject({
      schedulerVisible: false,
      sdVaeVisible: false,
    });
    expect(
      getGenerationModelPolicy(createModel('qwen-image'), createSettings(createModel('qwen-image'))).ui.schedulerVisible
    ).toBe(false);
    expect(getGenerationModelPolicy(externalModel, createSettings(externalModel)).ui.seedVisible).toBe(false);
    expect(
      getGenerationModelPolicy(
        { ...externalModel, capabilities: { supports_seed: true } },
        createSettings(externalModel)
      ).ui.seedVisible
    ).toBe(true);
  });

  it('has generation config for every graph builder base', () => {
    expect(SUPPORTED_GENERATE_BASES).toEqual([
      'sd-1',
      'sd-2',
      'sdxl',
      'sd-3',
      'flux',
      'flux2',
      'cogview4',
      'qwen-image',
      'z-image',
      'anima',
    ]);
  });

  it('does not mark display-only bases as generatable', () => {
    expect(isSupportedGenerateModel(createModel('sdxl-refiner'))).toBe(false);
    expect(isSupportedGenerateModel(createModel('unknown'))).toBe(false);
    expect(isSupportedGenerateModel(createModel('made-up'))).toBe(false);
    expect(BASE_GENERATION).not.toHaveProperty('external');
  });
});

describe('component policies', () => {
  it('validates FLUX T5, CLIP embed, and VAE requirements', () => {
    const model = createModel('flux');

    expect(getGenerationValidationReasons(model, createSettings(model))).toEqual([
      'Generate needs a T5 Encoder for FLUX models.',
      'Generate needs a CLIP Embed model for FLUX models.',
      'Generate needs a VAE for FLUX models.',
    ]);
    expect(
      getGenerationValidationReasons(
        model,
        createSettings(model, { clipEmbedModel: clipEmbed, t5EncoderModel: t5Encoder, vae: fluxVae })
      )
    ).toEqual([]);
    expect(
      getGenerationValidationReasons(
        model,
        createSettings(model, {
          clipEmbedModel: t5Encoder,
          t5EncoderModel: clipEmbed,
          vae: fluxVae,
        })
      )
    ).toEqual(['Generate needs a T5 Encoder for FLUX models.', 'Generate needs a CLIP Embed model for FLUX models.']);
  });

  it('rejects unsupported FLUX dev_fill variant', () => {
    const model = createModel('flux', { variant: 'dev_fill' });

    expect(
      getGenerationValidationReasons(
        model,
        createSettings(model, { clipEmbedModel: clipEmbed, t5EncoderModel: t5Encoder, vae: fluxVae })
      )
    ).toEqual(['FLUX Fill models do not support text-to-image generation.']);
  });

  it('validates FLUX.2 Qwen3 and VAE requirements and allows bundled alternatives', () => {
    const model = createModel('flux2', { format: 'gguf_quantized', variant: 'klein_9b' });
    const source = createModel('flux2', { format: 'diffusers', variant: 'klein_9b' });
    const incompatibleSource = createModel('flux2', {
      format: 'diffusers',
      key: 'flux2-4b-source',
      variant: 'klein_4b',
    });

    expect(getGenerationValidationReasons(model, createSettings(model))).toEqual([
      'Generate needs a Qwen3 Encoder for non-Diffusers FLUX.2 models.',
      'Generate needs a VAE for non-Diffusers FLUX.2 models.',
    ]);
    expect(getGenerationValidationReasons(model, createSettings(model, { componentSourceModel: source }))).toEqual([]);
    expect(
      getGenerationValidationReasons(model, createSettings(model, { componentSourceModel: incompatibleSource }))
    ).toEqual(['Generate needs a Qwen3 Encoder for non-Diffusers FLUX.2 models.']);
    expect(
      getGenerationValidationReasons(model, createSettings(model, { qwen3EncoderModel: qwen3Encoder, vae: flux2Vae }))
    ).toEqual([]);
  });

  it('keeps FLUX.2 component source hidden and auto-selects installed Diffusers sources', () => {
    const model = createModel('flux2', { format: 'gguf_quantized', variant: 'klein_9b' });
    const source = createModel('flux2', { format: 'diffusers', variant: 'klein_9b' });
    const incompatibleSource = createModel('flux2', {
      format: 'diffusers',
      key: 'flux2-4b-source',
      variant: 'klein_4b',
    });
    const settings = createSettings(model);

    expect(getComponentSectionPolicy(model, settings).slots.map((slot) => slot.key)).toEqual([
      'qwen3EncoderModel',
      'vae',
    ]);
    expect(getAutoFlux2ComponentSourceModel(model, settings, [incompatibleSource, source])?.key).toBe(source.key);
    expect(
      getAutoFlux2ComponentSourceModel(model, { ...settings, qwen3EncoderModel: qwen3Encoder }, [incompatibleSource])
        ?.key
    ).toBe(incompatibleSource.key);
  });

  it('validates Qwen Image Qwen-VL and VAE requirements', () => {
    const model = createModel('qwen-image', { format: 'checkpoint' });

    expect(getGenerationValidationReasons(model, createSettings(model))).toEqual([
      'Generate needs a Qwen VL Encoder for non-Diffusers Qwen Image models.',
      'Generate needs a VAE for non-Diffusers Qwen Image models.',
    ]);
    expect(
      getGenerationValidationReasons(
        model,
        createSettings(model, { qwenVLEncoderModel: qwenVlEncoder, vae: qwenImageVae })
      )
    ).toEqual([]);
  });

  it('validates Z-Image non-anima Qwen3 and flux VAE requirements', () => {
    const model = createModel('z-image', { format: 'checkpoint' });

    expect(getGenerationValidationReasons(model, createSettings(model))).toEqual([
      'Generate needs a Qwen3 Encoder for Z-Image models.',
      'Generate needs a VAE for Z-Image models.',
    ]);
    expect(
      getGenerationValidationReasons(model, createSettings(model, { qwen3EncoderModel: qwen3Encoder, vae: fluxVae }))
    ).toEqual([]);
  });

  it('validates Anima Qwen3 and allowed VAE bases', () => {
    const model = createModel('anima');

    expect(getGenerationValidationReasons(model, createSettings(model))).toEqual([
      'Generate needs a Qwen3 Encoder for Anima models.',
      'Generate needs a VAE for Anima models.',
    ]);
    expect(
      getGenerationValidationReasons(
        model,
        createSettings(model, { qwen3EncoderModel: animaQwen3Encoder, vae: qwenImageVae })
      )
    ).toEqual([]);
  });

  it('validates dimensions against bounds and model-family grid', () => {
    const model = createModel('flux2');

    expect(getGenerationValidationReasons(model, createSettings(model, { width: 0 }))).toContain(
      'Generate width must be between 64 and 4096.'
    );
    expect(getGenerationValidationReasons(model, createSettings(model, { height: 4104 }))).toContain(
      'Generate height must be between 64 and 4096.'
    );
    expect(getGenerationValidationReasons(model, createSettings(model, { height: 888 }))).toContain(
      'Generate height must be a multiple of 16.'
    );
  });

  it('rejects external image generators without a registered invocation node', () => {
    const model: GenerateModelConfig = {
      base: 'external',
      capabilities: { modes: ['txt2img'] },
      format: 'external_api',
      key: 'external-unknown',
      name: 'Unknown External Provider',
      provider_id: 'future-provider',
      type: 'external_image_generator',
    };

    expect(getGenerationValidationReasons(model, createSettings(model))).toContain(
      "No invocation node registered for external provider 'future-provider'."
    );
  });

  it('renders no component slots for bases without extra requirements', () => {
    expect(getComponentSectionPolicy(createModel('sdxl'), createSettings(createModel('sdxl'))).slots).toEqual([]);
    expect(getComponentSectionPolicy(createModel('cogview4'), createSettings(createModel('cogview4'))).slots).toEqual(
      []
    );
  });

  it('clears incompatible selections when the selected model changes', () => {
    const model = createModel('sdxl');
    const settings = createSettings(createModel('sd-1'), {
      cfgRescaleMultiplier: 0.5,
      clipSkip: 2,
      loras: [
        { isEnabled: true, model: sd1Lora, weight: 1 },
        { isEnabled: true, model: sdxlLora, weight: 1 },
      ],
      seamlessXAxis: true,
      t5EncoderModel: t5Encoder,
      vae: sdxlVae,
    });

    const result = getSettingsWithCompatibleModelSelections(settings, model);

    expect(result.settings.modelKey).toBe(model.key);
    expect(result.settings.loras.map((lora) => lora.model.key)).toEqual(['sdxl-lora']);
    expect(result.settings.vae).toEqual(sdxlVae);
    expect(result.settings.t5EncoderModel).toBeNull();
    expect(result.settings.clipSkip).toBe(0);
    expect(result.settings.cfgRescaleMultiplier).toBe(0);
    expect(result.settings.seamlessXAxis).toBe(true);
    expect(result.clearedLabels).toEqual(['LoRAs', 'T5 Encoder', 'CLIP skip', 'CFG rescale']);
  });

  it('reconciles dimensions to the new model grid when the selected model changes', () => {
    const model = createModel('cogview4');
    const settings = createSettings(createModel('sdxl'), { height: 520, width: 520 });
    const result = getSettingsWithCompatibleModelSelections(settings, model);

    expect(result.settings.width).toBe(512);
    expect(result.settings.height).toBe(512);
    expect(result.clearedLabels).toContain('Dimensions');
  });

  it('reports selected model records that disappeared from the backend model list', () => {
    const model = createModel('flux');
    const settings = createSettings(model, {
      clipEmbedModel: clipEmbed,
      loras: [{ isEnabled: true, model: sd1Lora, weight: 1 }],
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
    });

    expect(getGenerationModelAvailabilityReasons(model, settings, [model as never, t5Encoder as never])).toEqual([
      'CLIP Embed "CLIP Embed" is no longer installed.',
      'VAE "FLUX VAE" is no longer installed.',
      'LoRA "SD 1 LoRA" is no longer installed.',
    ]);
  });

  it('clears same-base FLUX.2 component selections that do not match the new variant', () => {
    const model = createModel('flux2', { format: 'gguf_quantized', variant: 'klein_9b' });
    const incompatibleQwen3: ComponentModelConfig = {
      base: 'any',
      key: 'qwen3-4b',
      name: 'Qwen3 4B Encoder',
      type: 'qwen3_encoder',
      variant: 'qwen3_4b',
    };
    const settings = createSettings(createModel('flux2', { format: 'gguf_quantized', variant: 'klein_4b' }), {
      qwen3EncoderModel: incompatibleQwen3,
      vae: flux2Vae,
    });

    const result = getSettingsWithCompatibleModelSelections(settings, model);

    expect(result.settings.qwen3EncoderModel).toBeNull();
    expect(result.settings.vae).toEqual(flux2Vae);
    expect(result.clearedLabels).toEqual(['Qwen3 Encoder']);
  });
});
