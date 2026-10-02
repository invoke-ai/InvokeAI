import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';
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
  coerceSchedulerForGraph,
  getComponentSectionPolicy,
  getAutoFlux2ComponentSourceModel,
  getDefaultGenerateSettings,
  getGenerateModelSelectionResult,
  getGenerationDimensions,
  getGenerationModelAvailabilityReasons,
  getGenerationModelPolicy,
  getGenerationValidationReasons,
  getMaxReferenceImages,
  getPromptPolicy,
  getSettingsWithModelDefaults,
  isReferenceImageSupported,
  isGenerateModelSelectable,
  isSupportedGenerateModel,
} from './baseGenerationPolicies';
import { SUPPORTED_GENERATE_BASES } from './supportedBases';

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
  seedMode: 'fixed',
  ...overrides,
});

const t5Encoder: ComponentModelConfig = { base: 'any', key: 't5', name: 'T5 Encoder', type: 't5_encoder' };
const clipEmbed: ComponentModelConfig = { base: 'any', key: 'clip', name: 'CLIP Embed', type: 'clip_embed' };
const mistralEncoder: ComponentModelConfig = {
  base: 'any',
  key: 'mistral',
  name: 'Mistral Encoder',
  type: 'mistral_encoder',
};
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
const sd1IpAdapter: ComponentModelConfig = {
  base: 'sd-1',
  key: 'sd1-ip-adapter',
  name: 'SD 1 IP Adapter',
  type: 'ip_adapter',
};
const sdxlIpAdapter: ComponentModelConfig = {
  base: 'sdxl',
  key: 'sdxl-ip-adapter',
  name: 'SDXL IP Adapter',
  type: 'ip_adapter',
};
const referenceImage = {
  original: { image: { height: 768, image_name: 'reference.png', width: 512 } },
};
const externalModel: GenerateModelConfig = {
  base: 'external',
  capabilities: { modes: ['txt2img'], supports_negative_prompt: false, supports_seed: false },
  format: 'external_api',
  key: 'external-model',
  name: 'External Model',
  provider_id: 'openai',
  type: 'external_image_generator',
};

seedArchitectureCapabilities();

describe('architecture policy, read from the backend capability table', () => {
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
    // The FLUX base row describes dev.
    expect(getDefaultGenerateSettings(createModel('flux'))).toMatchObject({
      steps: 28,
      cfgScale: 3.5,
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
    // Tongyi-MAI/Z-Image-Turbo says num_inference_steps=9.
    expect(getDefaultGenerateSettings(createModel('z-image'))).toMatchObject({
      steps: 9,
      cfgScale: 1,
      scheduler: 'euler',
    });
  });

  it('answers per variant where the architecture does', () => {
    expect(getDefaultGenerateSettings(createModel('flux', { variant: 'schnell' }))).toMatchObject({ steps: 4 });
    expect(getDefaultGenerateSettings(createModel('flux', { variant: 'dev_fill' }))).toMatchObject({
      steps: 50,
      cfgScale: 30,
    });
    expect(getDefaultGenerateSettings(createModel('z-image', { variant: 'zbase' }))).toMatchObject({
      steps: 50,
      cfgScale: 4,
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
    expect(getPromptPolicy(createModel('flux'), { cfgScale: 4, negativePromptEnabled: true })).toMatchObject({
      negativeVisible: false,
      negativeUsedInGraph: false,
    });
    expect(getPromptPolicy(createModel('sdxl'), { cfgScale: 1, negativePromptEnabled: true })).toMatchObject({
      negativeVisible: true,
      negativeUsedInGraph: true,
    });
    expect(getPromptPolicy(createModel('sdxl'), { cfgScale: 1, negativePromptEnabled: false })).toMatchObject({
      negativeVisible: true,
      negativeUsedInGraph: false,
    });
    expect(getPromptPolicy(createModel('qwen-image'), { cfgScale: 1, negativePromptEnabled: true })).toMatchObject({
      negativeVisible: true,
      negativeUsedInGraph: false,
    });
    expect(getPromptPolicy(createModel('qwen-image'), { cfgScale: 2, negativePromptEnabled: true })).toMatchObject({
      negativeVisible: true,
      negativeUsedInGraph: true,
      negativeHelpText: 'Used only when CFG is greater than 1.',
    });
    expect(getPromptPolicy(externalModel, { cfgScale: 7, negativePromptEnabled: true })).toMatchObject({
      negativeVisible: false,
      negativeUsedInGraph: false,
    });
    expect(
      getPromptPolicy(
        { ...externalModel, capabilities: { supports_negative_prompt: true } },
        { cfgScale: 7, negativePromptEnabled: true }
      )
    ).toMatchObject({ negativeVisible: true, negativeUsedInGraph: true });
  });

  it('matches expected UI availability per base', () => {
    expect(getGenerationModelPolicy(createModel('sd-1'), createSettings(createModel('sd-1'))).ui).toMatchObject({
      clipSkipMax: 12,
      cfgRescaleVisible: true,
      hiDiffusionVisible: true,
      schedulerVisible: true,
    });
    expect(getGenerationModelPolicy(createModel('sd-2'), createSettings(createModel('sd-2'))).ui).toMatchObject({
      clipSkipMax: 24,
      cfgRescaleVisible: true,
      hiDiffusionVisible: false,
    });
    expect(getGenerationModelPolicy(createModel('sdxl'), createSettings(createModel('sdxl'))).ui).toMatchObject({
      hiDiffusionVisible: true,
    });
    expect(getGenerationModelPolicy(createModel('flux'), createSettings(createModel('flux'))).ui).toMatchObject({
      guidanceLabel: 'Guidance',
      hiDiffusionVisible: false,
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

  // The ordered list of generatable bases is pinned in `supportedBases.test.ts`, which now owns it.

  it('does not mark display-only bases as generatable', () => {
    expect(isSupportedGenerateModel(createModel('sdxl-refiner'))).toBe(false);
    expect(isSupportedGenerateModel(createModel('unknown'))).toBe(false);
    expect(isSupportedGenerateModel(createModel('made-up'))).toBe(false);
    // Handle external pseudo-bases through model type, not architecture rows.
    expect(isSupportedGenerateModel(createModel('external'))).toBe(false);
  });

  // Read the way the architecture suite reads sources; `node:fs` is not typed for this project.
  const SOURCES = import.meta.glob(
    [
      '../../../workbench/palette/paletteProviders.ts',
      '../ui/GenerateWidgetView.tsx',
      '../ui/GenerateModelCard.tsx',
      './resolveGenerateWidgetValues.ts',
    ],
    { eager: true, import: 'default', query: '?raw' }
  ) as Record<string, string>;

  describe('isGenerateModelSelectable', () => {
    const unconditional = createModel('ideogram-4', { branch: 'unconditional', format: 'checkpoint' });
    const conditional = createModel('ideogram-4', { branch: 'conditional', format: 'checkpoint' });

    it('keeps a model that cannot run on its own out of every picker', () => {
      // An unconditional Ideogram branch is not independently generatable despite its main type.
      expect(isSupportedGenerateModel(unconditional)).toBe(true);
      expect(isGenerateModelSelectable(unconditional)).toBe(false);

      expect(isGenerateModelSelectable(conditional)).toBe(true);
      expect(isGenerateModelSelectable(createModel('ideogram-4', { format: 'diffusers' }))).toBe(true);
      // Wan's low-noise expert remains selectable because the other expert can span the schedule.
      expect(isGenerateModelSelectable(createModel('wan', { variant: 'a14b' }))).toBe(true);
    });

    it('is what every path that offers or picks a model filters with', () => {
      // Check actual selection calls across entry points, not merely their imports.
      const callSites = {
        '../../../workbench/palette/paletteProviders.ts': '.filter(isGenerateModelSelectable)',
        './resolveGenerateWidgetValues.ts': 'models.filter(isGenerateModelSelectable)',
        '../ui/GenerateWidgetView.tsx': 'models.filter(isGenerateModelSelectable)',
        '../ui/GenerateModelCard.tsx': 'isGenerateModelSelectable(model)',
      };

      for (const [path, call] of Object.entries(callSites)) {
        const source = SOURCES[path];

        expect(source, `${path} is not in the source glob`).toBeTypeOf('string');
        expect(source, `${path} must select models with isGenerateModelSelectable`).toContain(call);
      }
    });
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

  it('requires standalone components for partial SDNQ folders but not complete pipelines', () => {
    const completePipeline = { text_encoder: {}, tokenizer: {}, transformer: {}, vae: {} };
    const completeFlux1Pipeline = { ...completePipeline, text_encoder_2: {}, tokenizer_2: {} };

    for (const model of [
      createModel('flux', { format: 'sdnq_quantized', submodels: completeFlux1Pipeline }),
      createModel('flux2', { format: 'sdnq_quantized', submodels: completePipeline, variant: 'klein_9b' }),
      createModel('z-image', { format: 'sdnq_quantized', submodels: completePipeline }),
    ]) {
      expect(getGenerationValidationReasons(model, createSettings(model))).toEqual([]);
    }

    expect(
      getGenerationValidationReasons(
        createModel('flux', { format: 'sdnq_quantized', submodels: { transformer: {} } }),
        createSettings(createModel('flux', { format: 'sdnq_quantized', submodels: { transformer: {} } }))
      )
    ).toContain('Generate needs a T5 Encoder for FLUX models.');

    for (const model of [
      createModel('flux2', { format: 'sdnq_quantized', submodels: { transformer: {} }, variant: 'klein_9b' }),
      createModel('z-image', { format: 'sdnq_quantized', submodels: { transformer: {} } }),
    ]) {
      expect(getGenerationValidationReasons(model, createSettings(model))).not.toEqual([]);
    }
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

  it('validates FLUX.2 [dev] Mistral and VAE requirements and allows bundled alternatives', () => {
    const model = createModel('flux2', { format: 'gguf_quantized', variant: 'dev' });
    const source = createModel('flux2', { format: 'diffusers', key: 'flux2-dev-source', variant: 'dev' });
    const kleinSource = createModel('flux2', {
      format: 'diffusers',
      key: 'flux2-klein-source',
      variant: 'klein_9b',
    });

    expect(getGenerationValidationReasons(model, createSettings(model))).toEqual([
      'Generate needs a Mistral Encoder for non-Diffusers FLUX.2 [dev] models.',
      'Generate needs a VAE for non-Diffusers FLUX.2 models.',
    ]);
    expect(getGenerationValidationReasons(model, createSettings(model, { componentSourceModel: source }))).toEqual([]);
    expect(getGenerationValidationReasons(model, createSettings(model, { componentSourceModel: kleinSource }))).toEqual(
      ['Generate needs a Mistral Encoder for non-Diffusers FLUX.2 [dev] models.']
    );
    expect(
      getGenerationValidationReasons(
        model,
        createSettings(model, { mistralEncoderModel: mistralEncoder, vae: flux2Vae })
      )
    ).toEqual([]);
    expect(getComponentSectionPolicy(model, createSettings(model)).slots.map((slot) => slot.key)).toEqual([
      'mistralEncoderModel',
      'vae',
      'pidDecoderModel',
      'gemma2EncoderModel',
    ]);
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

    // FLUX.2 supports PiD, so its two slots follow the base's own components.
    expect(getComponentSectionPolicy(model, settings).slots.map((slot) => slot.key)).toEqual([
      'qwen3EncoderModel',
      'vae',
      'pidDecoderModel',
      'gemma2EncoderModel',
    ]);
    expect(getAutoFlux2ComponentSourceModel(model, settings, [incompatibleSource, source])?.key).toBe(source.key);
    expect(
      getAutoFlux2ComponentSourceModel(model, { ...settings, qwen3EncoderModel: qwen3Encoder }, [incompatibleSource])
        ?.key
    ).toBe(incompatibleSource.key);
  });

  it('auto-selects complete FLUX.2 SDNQ component sources and rejects partial folders', () => {
    const model = createModel('flux2', { format: 'gguf_quantized', variant: 'klein_9b' });
    const completeSource = createModel('flux2', {
      format: 'sdnq_quantized',
      key: 'flux2-complete-sdnq-source',
      submodels: { text_encoder: {}, tokenizer: {}, transformer: {}, vae: {} },
      variant: 'klein_9b',
    });
    const partialSource = createModel('flux2', {
      format: 'sdnq_quantized',
      key: 'flux2-partial-sdnq-source',
      submodels: { text_encoder: {}, transformer: {}, vae: {} },
      variant: 'klein_9b',
    });
    const settings = createSettings(model);

    expect(getAutoFlux2ComponentSourceModel(model, settings, [partialSource, completeSource])?.key).toBe(
      completeSource.key
    );
    expect(getAutoFlux2ComponentSourceModel(model, settings, [partialSource])).toBeNull();
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

  it('offers complete SDNQ Z-Image pipelines as component sources but rejects partial folders', () => {
    const model = createModel('z-image', { format: 'checkpoint' });
    const settings = createSettings(model);
    const sourceFilter = getComponentSectionPolicy(model, settings).slots.find(
      (slot) => slot.key === 'componentSourceModel'
    )?.filter;
    const complete = createModel('z-image', {
      format: 'sdnq_quantized',
      submodels: { text_encoder: {}, tokenizer: {}, transformer: {}, vae: {} },
    });
    const partial = createModel('z-image', {
      format: 'sdnq_quantized',
      submodels: { transformer: {} },
    });

    expect(sourceFilter?.(complete, { model, selectedComponents: { ...settings }, settings })).toBe(true);
    expect(sourceFilter?.(partial, { model, selectedComponents: { ...settings }, settings })).toBe(false);
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
    expect(getComponentSectionPolicy(createModel('cogview4'), createSettings(createModel('cogview4'))).slots).toEqual(
      []
    );
  });

  it('offers only the PiD slots for a PiD-capable base with no other components', () => {
    // SDXL exposes PiD slots before enabling PiD; they become required only when enabled.
    const model = createModel('sdxl');

    expect(getComponentSectionPolicy(model, createSettings(model)).slots.map((slot) => slot.key)).toEqual([
      'pidDecoderModel',
      'gemma2EncoderModel',
    ]);
  });

  it('does not offer PiD slots for a base with no PiD decode node', () => {
    const model = createModel('cogview4');

    expect(getComponentSectionPolicy(model, createSettings(model)).slots).toEqual([]);
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

    const result = getGenerateModelSelectionResult({ currentValues: settings, model, models: [] });

    expect(result.settings.modelKey).toBe(model.key);
    expect(result.settings.loras.map((lora) => lora.model.key)).toEqual(['sdxl-lora']);
    expect(result.settings.vae).toEqual(sdxlVae);
    expect(result.settings.t5EncoderModel).toBeNull();
    expect(result.settings.clipSkip).toBe(0);
    expect(result.settings.cfgRescaleMultiplier).toBe(0);
    expect(result.settings.seamlessXAxis).toBe(true);
    expect(result.clearedLabels).toEqual(['LoRAs', 'T5 Encoder', 'CLIP skip', 'CFG rescale']);
  });

  it('turns HiDiffusion off when changing to an unsupported model family', () => {
    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sd-1'), { hiDiffusionEnabled: true }),
      model: createModel('flux'),
      models: [],
    });

    expect(result.settings.hiDiffusionEnabled).toBe(false);
    expect(result.clearedLabels).toContain('HiDiffusion');
  });

  it('initializes empty and malformed values from the selected model defaults', () => {
    const model = createModel('cogview4', { default_settings: { steps: 17 } });

    for (const currentValues of [{}, { batchCount: 8 }]) {
      const result = getGenerateModelSelectionResult({ currentValues, model, models: [] });

      expect(result.settings).toMatchObject({
        batchCount: 1,
        height: 1024,
        modelKey: model.key,
        steps: 17,
        width: 1024,
      });
      expect(result.clearedLabels).toEqual([]);
    }
  });

  it('preserves model-independent and compatible settings across a model selection', () => {
    const model = createModel('sdxl', { key: 'sdxl-other' });
    const settings = createSettings(createModel('sdxl'), {
      batchCount: 4,
      negativePrompt: 'blurry',
      positivePrompt: 'a lighthouse',
      seed: 1234,
      seedMode: 'fixed',
      vae: sdxlVae,
    });

    const result = getGenerateModelSelectionResult({ currentValues: settings, model, models: [] });

    expect(result.settings).toMatchObject({
      batchCount: 4,
      modelKey: model.key,
      negativePrompt: 'blurry',
      positivePrompt: 'a lighthouse',
      seed: 1234,
      seedMode: 'fixed',
      vae: sdxlVae,
    });
    expect(result.clearedLabels).toEqual([]);
  });

  it('applies automatic FLUX.2 component sources as part of model selection', () => {
    const model = createModel('flux2', { format: 'gguf_quantized', variant: 'klein_9b' });
    const source = createModel('flux2', { format: 'diffusers', key: 'flux2-source', variant: 'klein_9b' });

    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sdxl')),
      model,
      models: [source],
    });

    expect(result.settings.componentSourceModel).toBe(source);
  });

  it('auto-selects a matching FLUX.2 [dev] source and rejects Klein sources', () => {
    const model = createModel('flux2', { format: 'gguf_quantized', variant: 'dev' });
    const devSource = createModel('flux2', { format: 'diffusers', key: 'flux2-dev-source', variant: 'dev' });
    const kleinSource = createModel('flux2', { format: 'diffusers', key: 'flux2-klein-source', variant: 'klein_9b' });

    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sdxl')),
      model,
      models: [kleinSource, devSource],
    });

    expect(result.settings.componentSourceModel).toBe(devSource);
  });

  it('reconciles dimensions to the new model grid when the selected model changes', () => {
    const model = createModel('cogview4');
    const settings = createSettings(createModel('sdxl'), { height: 520, width: 520 });
    const result = getGenerateModelSelectionResult({ currentValues: settings, model, models: [] });

    expect(result.settings.width).toBe(512);
    expect(result.settings.height).toBe(512);
    expect(result.clearedLabels).toContain('Dimensions');
  });

  it('keeps selected VAE when applying model defaults without a VAE override', () => {
    const model = createModel('sdxl', { default_settings: { steps: 12 } });
    const settings = createSettings(model, { vae: sdxlVae });
    const nextSettings = getSettingsWithModelDefaults(settings, model);

    expect(nextSettings.steps).toBe(12);
    expect(nextSettings.vae).toBe(sdxlVae);
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

    const result = getGenerateModelSelectionResult({ currentValues: settings, model, models: [] });

    expect(result.settings.qwen3EncoderModel).toBeNull();
    expect(result.settings.vae).toEqual(flux2Vae);
    expect(result.clearedLabels).toEqual(['Qwen3 Encoder']);
  });

  it('clears Klein and dev encoders when switching FLUX.2 variants', () => {
    const devModel = createModel('flux2', { format: 'gguf_quantized', variant: 'dev' });
    const kleinModel = createModel('flux2', { format: 'gguf_quantized', variant: 'klein_9b' });
    const toDev = getGenerateModelSelectionResult({
      currentValues: createSettings(kleinModel, { qwen3EncoderModel: qwen3Encoder, vae: flux2Vae }),
      model: devModel,
      models: [],
    });
    const toKlein = getGenerateModelSelectionResult({
      currentValues: createSettings(devModel, { mistralEncoderModel: mistralEncoder, vae: flux2Vae }),
      model: kleinModel,
      models: [],
    });

    expect(toDev.settings.qwen3EncoderModel).toBeNull();
    expect(toDev.clearedLabels).toContain('Qwen3 Encoder');
    expect(toKlein.settings.mistralEncoderModel).toBeNull();
    expect(toKlein.clearedLabels).toContain('Mistral Encoder');
  });

  it('requires enabled IP Adapter reference images to have an image and compatible adapter model', () => {
    const model = createModel('sd-1');

    expect(
      getGenerationValidationReasons(
        model,
        createSettings(model, {
          referenceImages: [
            {
              id: 'ref-1',
              isEnabled: true,
              config: {
                beginEndStepPct: [0, 1],
                clipVisionModel: 'ViT-H',
                image: null,
                method: 'full',
                model: sd1IpAdapter,
                type: 'ip_adapter',
                weight: 1,
              },
            },
          ],
        })
      )
    ).toContain('Reference Image #1 needs an image.');

    expect(
      getGenerationValidationReasons(
        model,
        createSettings(model, {
          referenceImages: [
            {
              id: 'ref-1',
              isEnabled: true,
              config: {
                beginEndStepPct: [0, 1],
                clipVisionModel: 'ViT-H',
                image: referenceImage,
                method: 'full',
                model: null,
                type: 'ip_adapter',
                weight: 1,
              },
            },
          ],
        })
      )
    ).toContain('Reference Image #1 needs a compatible IP Adapter model.');
  });

  it('clears reference images when switching to a model without reference image support', () => {
    const settings = createSettings(createModel('sdxl'), {
      referenceImages: [
        {
          id: 'ref-1',
          isEnabled: true,
          config: {
            beginEndStepPct: [0, 1],
            clipVisionModel: 'ViT-H',
            image: referenceImage,
            method: 'full',
            model: sdxlIpAdapter,
            type: 'ip_adapter',
            weight: 1,
          },
        },
      ],
    });

    const result = getGenerateModelSelectionResult({
      currentValues: settings,
      model: createModel('cogview4'),
      models: [sdxlIpAdapter],
    });

    expect(result.settings.referenceImages).toEqual([]);
    expect(result.clearedLabels).toContain('Reference Images');
  });

  it('re-targets reference image configs to the new model base, keeping the image and enabled state', () => {
    const settings = createSettings(createModel('sdxl'), {
      referenceImages: [
        {
          id: 'ref-1',
          isEnabled: false,
          config: {
            beginEndStepPct: [0, 1],
            clipVisionModel: 'ViT-H',
            image: referenceImage,
            method: 'style',
            model: sdxlIpAdapter,
            type: 'ip_adapter',
            weight: 0.5,
          },
        },
      ],
    });

    const result = getGenerateModelSelectionResult({
      currentValues: settings,
      model: createModel('sd-1'),
      models: [sd1IpAdapter, sdxlIpAdapter],
    });

    expect(result.settings.referenceImages).toEqual([
      {
        id: 'ref-1',
        isEnabled: false,
        config: {
          beginEndStepPct: [0, 1],
          clipVisionModel: 'ViT-H',
          image: referenceImage,
          method: 'full',
          model: sd1IpAdapter,
          type: 'ip_adapter',
          weight: 1,
        },
      },
    ]);
    expect(result.clearedLabels).toContain('Reference Images');
  });

  it('re-targets reference images across a flux2 -> sdxl -> flux2 round trip', () => {
    const flux2Model = createModel('flux2');
    const sdxlModel = createModel('sdxl');
    const settings = createSettings(flux2Model, {
      referenceImages: [
        { id: 'ref-1', isEnabled: true, config: { image: referenceImage, type: 'flux2_reference_image' } },
      ],
    });

    const toSdxl = getGenerateModelSelectionResult({
      currentValues: settings,
      model: sdxlModel,
      models: [sdxlIpAdapter],
    });

    expect(toSdxl.settings.referenceImages[0]?.config).toMatchObject({
      image: referenceImage,
      model: sdxlIpAdapter,
      type: 'ip_adapter',
    });

    const backToFlux2 = getGenerateModelSelectionResult({
      currentValues: toSdxl.settings,
      model: flux2Model,
      models: [sdxlIpAdapter],
    });

    expect(backToFlux2.settings.referenceImages).toEqual([
      { id: 'ref-1', isEnabled: true, config: { image: referenceImage, type: 'flux2_reference_image' } },
    ]);
  });

  it('keeps compatible reference images untouched when the selected model changes', () => {
    const referenceImages: GenerateSettings['referenceImages'] = [
      {
        id: 'ref-1',
        isEnabled: true,
        config: {
          beginEndStepPct: [0, 1],
          clipVisionModel: 'ViT-H',
          image: referenceImage,
          method: 'full',
          model: sdxlIpAdapter,
          type: 'ip_adapter',
          weight: 1,
        },
      },
    ];
    const settings = createSettings(createModel('sdxl'), { referenceImages });

    const result = getGenerateModelSelectionResult({
      currentValues: settings,
      model: createModel('sdxl', { key: 'sdxl-other' }),
      models: [sdxlIpAdapter],
    });

    expect(result.settings.referenceImages).toBe(referenceImages);
    expect(result.clearedLabels).not.toContain('Reference Images');
  });

  it('reports reference image support per model', () => {
    expect(isReferenceImageSupported(createModel('sdxl'))).toBe(true);
    expect(isReferenceImageSupported(createModel('cogview4'))).toBe(false);
    expect(isReferenceImageSupported(createModel('qwen-image'))).toBe(false);
    expect(isReferenceImageSupported(createModel('qwen-image', { variant: 'edit' }))).toBe(true);
    expect(isReferenceImageSupported(externalModel)).toBe(false);
    expect(
      isReferenceImageSupported({
        ...externalModel,
        capabilities: { ...externalModel.capabilities, supports_reference_images: true },
      })
    ).toBe(true);
    expect(isReferenceImageSupported(undefined)).toBe(false);
  });

  it('derives the reference image limit from external provider capabilities', () => {
    expect(getMaxReferenceImages(createModel('sdxl'))).toBe(5);
    expect(getMaxReferenceImages(createModel('cogview4'))).toBe(0);
    expect(
      getMaxReferenceImages({
        ...externalModel,
        capabilities: { ...externalModel.capabilities, max_reference_images: 14, supports_reference_images: true },
      })
    ).toBe(14);
    expect(
      getMaxReferenceImages({
        ...externalModel,
        capabilities: { ...externalModel.capabilities, supports_reference_images: true },
      })
    ).toBe(5);
  });

  it('validates the reference image count against the served limit, not a local default', () => {
    // A nondefault served limit distinguishes table reads from hardcoded policy.
    setArchitectureCapabilities(
      architectureCapabilitiesFixture.map((row) =>
        row.base === 'sdxl' ? { ...row, features: { ...row.features, max_reference_images: 2 } } : row
      )
    );
    const model = createModel('sdxl');
    const ipAdapterReference = (index: number) => ({
      config: {
        beginEndStepPct: [0, 1] as [number, number],
        clipVisionModel: 'ViT-H' as const,
        image: referenceImage,
        method: 'full' as const,
        model: sdxlIpAdapter,
        type: 'ip_adapter' as const,
        weight: 1,
      },
      id: `ref-${index}`,
      isEnabled: true,
    });
    const countReasons = (count: number) =>
      getGenerationValidationReasons(
        model,
        createSettings(model, {
          referenceImages: Array.from({ length: count }, (_, index) => ipAdapterReference(index)),
        })
      ).filter((reason) => reason.startsWith('Generate supports at most'));

    expect(countReasons(2)).toEqual([]);
    expect(countReasons(3)).toEqual(['Generate supports at most 2 reference images for sdxl model.']);
  });

  it('rejects unsupported reference image configs for the selected model', () => {
    const model = createModel('sd-3');

    expect(
      getGenerationValidationReasons(
        model,
        createSettings(model, {
          referenceImages: [
            { id: 'ref-1', isEnabled: true, config: { image: referenceImage, type: 'flux2_reference_image' } },
          ],
        })
      )
    ).toContain('Reference Image #1 is not supported by sd-3 model.');
  });

  it('reports missing reference image model dependencies', () => {
    const model = createModel('sd-1');

    expect(
      getGenerationModelAvailabilityReasons(
        model,
        createSettings(model, {
          referenceImages: [
            {
              id: 'ref-1',
              isEnabled: true,
              config: {
                beginEndStepPct: [0, 1],
                clipVisionModel: 'ViT-H',
                image: referenceImage,
                method: 'full',
                model: sd1IpAdapter,
                type: 'ip_adapter',
                weight: 1,
              },
            },
          ],
        }),
        [model as never]
      )
    ).toContain('Reference Image model "SD 1 IP Adapter" is no longer installed.');
  });

  it('ignores missing reference image model dependencies when the reference image is disabled', () => {
    const model = createModel('sd-1');

    expect(
      getGenerationModelAvailabilityReasons(
        model,
        createSettings(model, {
          referenceImages: [
            {
              id: 'ref-1',
              isEnabled: false,
              config: {
                beginEndStepPct: [0, 1],
                clipVisionModel: 'ViT-H',
                image: referenceImage,
                method: 'full',
                model: sd1IpAdapter,
                type: 'ip_adapter',
                weight: 1,
              },
            },
          ],
        }),
        [model as never]
      )
    ).toEqual([]);
  });
});

describe('Krea-2, Ideogram 4 and Wan policies', () => {
  const krea2Vae: VaeModelConfig = { base: 'qwen-image', key: 'krea2-vae', name: 'Qwen VAE', type: 'vae' };
  const qwen3VlEncoder: ComponentModelConfig = {
    base: 'any',
    key: 'qwen3-vl',
    name: 'Qwen3-VL',
    type: 'qwen3_vl_encoder',
    variant: 'qwen3_vl_4b',
  };
  const wanVae: VaeModelConfig = { base: 'wan', key: 'wan-vae', latent_channels: 16, name: 'Wan VAE', type: 'vae' };
  const wanT5Encoder: ComponentModelConfig = { base: 'any', key: 'wan-t5', name: 'Wan T5', type: 'wan_t5_encoder' };

  it('uses a 16px grid for all three, matching their transformer patch sizes', () => {
    expect(getGenerationDimensions(createModel('krea-2'))).toMatchObject({ grid: 16, optimal: 1024 });
    expect(getGenerationDimensions(createModel('ideogram-4'))).toMatchObject({ grid: 16, optimal: 1024 });
    expect(getGenerationDimensions(createModel('wan'))).toMatchObject({ grid: 16, optimal: 1024 });
  });

  it('rejects Ideogram 4 dimensions that are not multiples of 16', () => {
    const model = createModel('ideogram-4');
    const reasons = getGenerationValidationReasons(model, createSettings(model, { height: 1020, width: 1000 }));

    expect(reasons).toContain('Generate width must be a multiple of 16.');
    expect(reasons).toContain('Generate height must be a multiple of 16.');
  });

  it('rejects Ideogram 4 step and mu overrides outside what its denoise node accepts', () => {
    // Use valid bounds and a Diffusers bundle to isolate scalar validation from missing components.
    const model = createModel('ideogram-4', { format: 'diffusers' });

    expect(getGenerationValidationReasons(model, createSettings(model, { ideogram4Steps: 1 }))).toContain(
      'Ideogram 4 steps must be between 2 and 100.'
    );
    expect(getGenerationValidationReasons(model, createSettings(model, { ideogram4Mu: 5 }))).toContain(
      'Ideogram 4 mu must be between -4 and 4.'
    );
    expect(
      getGenerationValidationReasons(model, createSettings(model, { ideogram4Mu: -4, ideogram4Steps: 2 }))
    ).toEqual([]);
  });

  it('rejects an Ideogram 4 guidance override outside what its denoise node accepts', () => {
    // Finite persisted Ideogram values still need range validation.
    const model = createModel('ideogram-4');
    const message = 'Ideogram 4 guidance must be between 1 and 20.';

    expect(getGenerationValidationReasons(model, createSettings(model, { ideogram4GuidanceScale: 0.5 }))).toContain(
      message
    );
    expect(getGenerationValidationReasons(model, createSettings(model, { ideogram4GuidanceScale: 20.5 }))).toContain(
      message
    );
    expect(getGenerationValidationReasons(model, createSettings(model, { ideogram4GuidanceScale: 5 }))).not.toContain(
      message
    );
    // Null means "let the preset decide" and is omitted from the graph entirely.
    expect(
      getGenerationValidationReasons(model, createSettings(model, { ideogram4GuidanceScale: null }))
    ).not.toContain(message);
  });

  it('accepts a Qwen-Image VAE registered under the anima base, as the backend loader does', () => {
    // Accept the same Qwen VAE under anima and qwen-image registrations.
    const animaRegisteredVae: VaeModelConfig = {
      base: 'anima',
      key: 'anima-qwen-vae',
      name: 'Qwen VAE (installed for Anima)',
      type: 'vae',
    };
    const checkpoint = createModel('krea-2', { format: 'checkpoint' });
    const settings = createSettings(checkpoint, {
      qwen3VLEncoderModel: qwen3VlEncoder,
      vae: animaRegisteredVae,
    });

    expect(getGenerationValidationReasons(checkpoint, settings)).toEqual([]);

    const slots = getComponentSectionPolicy(checkpoint, settings).slots;
    const vaeSlotPolicy = slots.find((slot) => slot.key === 'vae');

    // Picker and validator must share compatibility rules.
    expect(
      vaeSlotPolicy?.filter?.(animaRegisteredVae, {
        model: checkpoint,
        selectedComponents: { ...settings },
        settings,
      })
    ).toBe(true);
  });

  it('still rejects a VAE from an unrelated family for Krea-2', () => {
    const checkpoint = createModel('krea-2', { format: 'checkpoint' });
    const reasons = getGenerationValidationReasons(
      checkpoint,
      createSettings(checkpoint, { qwen3VLEncoderModel: qwen3VlEncoder, vae: sdxlVae })
    );

    expect(reasons).toContain('Generate needs a VAE for non-Diffusers Krea-2 models.');
  });

  it('requires a VAE and Qwen3-VL encoder for a non-diffusers Krea-2 but not a diffusers one', () => {
    const checkpoint = createModel('krea-2', { format: 'checkpoint' });
    const missing = getGenerationValidationReasons(checkpoint, createSettings(checkpoint));

    expect(missing).toContain('Generate needs a VAE for non-Diffusers Krea-2 models.');
    expect(missing).toContain('Generate needs a Qwen3-VL Encoder for non-Diffusers Krea-2 models.');

    const supplied = getGenerationValidationReasons(
      checkpoint,
      createSettings(checkpoint, { qwen3VLEncoderModel: qwen3VlEncoder, vae: krea2Vae })
    );

    expect(supplied).toEqual([]);

    const diffusers = createModel('krea-2', { format: 'diffusers' });

    expect(getGenerationValidationReasons(diffusers, createSettings(diffusers))).toEqual([]);
  });

  it('blocks unparseable Krea-2 rebalance weights only while rebalance is on', () => {
    const model = createModel('krea-2', { format: 'diffusers' });
    const invalid = { krea2RebalanceEnabled: true, krea2RebalanceWeights: '1.0,2.0' };

    expect(getGenerationValidationReasons(model, createSettings(model, invalid))).toContain(
      'Krea-2 rebalance weights must be 12 comma-separated numbers.'
    );
    // Off, the string never reaches the backend parser, so it must not block generation.
    expect(
      getGenerationValidationReasons(model, createSettings(model, { ...invalid, krea2RebalanceEnabled: false }))
    ).toEqual([]);
  });

  it('rejects non-numeric Krea-2 weights even at the right count', () => {
    const model = createModel('krea-2', { format: 'diffusers' });
    const weights = '1,1,1,1,1,1,1,1,1,1,1,nope';

    expect(
      getGenerationValidationReasons(
        model,
        createSettings(model, { krea2RebalanceEnabled: true, krea2RebalanceWeights: weights })
      )
    ).toContain('Krea-2 rebalance weights must be 12 comma-separated numbers.');
  });

  it('requires a VAE and Wan T5 encoder for a GGUF Wan main', () => {
    const gguf = createModel('wan', { format: 'gguf_quantized' });
    const missing = getGenerationValidationReasons(gguf, createSettings(gguf));

    expect(missing).toContain('Generate needs a VAE for Wan models.');
    expect(missing).toContain('Generate needs a Wan T5 Encoder for Wan models.');

    expect(
      getGenerationValidationReasons(gguf, createSettings(gguf, { vae: wanVae, wanT5EncoderModel: wanT5Encoder }))
    ).toEqual([]);
  });

  it('never requires the low-noise Wan expert, which is an optional quality upgrade', () => {
    const diffusers = createModel('wan', { format: 'diffusers' });

    expect(getGenerationValidationReasons(diffusers, createSettings(diffusers))).toEqual([]);
  });

  it('reports an enabled Wan LoRA that targets the other family instead of dropping it silently', () => {
    const main = createModel('wan', { format: 'diffusers', variant: 'ti2v_5b' });
    const a14bLora: LoraModelConfig = {
      base: 'wan',
      key: 'a14b-lora',
      name: 'A14B LoRA',
      type: 'lora',
      variant: 'a14b',
    };
    const reasons = getGenerationValidationReasons(
      main,
      createSettings(main, { loras: [{ isEnabled: true, model: a14bLora, weight: 1 }] })
    );

    expect(reasons).toContain(`A14B LoRA targets a different Wan model family than ${main.name}.`);
  });

  it('accepts a Wan LoRA whose family matches, across both A14B main variants', () => {
    const a14bLora: LoraModelConfig = {
      base: 'wan',
      key: 'a14b-lora',
      name: 'A14B LoRA',
      type: 'lora',
      variant: 'a14b',
    };

    for (const variant of ['t2v_a14b', 'i2v_a14b']) {
      const main = createModel('wan', { format: 'diffusers', variant });

      expect(
        getGenerationValidationReasons(
          main,
          createSettings(main, { loras: [{ isEnabled: true, model: a14bLora, weight: 1 }] })
        )
      ).toEqual([]);
    }
  });

  it('ignores a disabled mismatched Wan LoRA', () => {
    const main = createModel('wan', { format: 'diffusers', variant: 'ti2v_5b' });
    const a14bLora: LoraModelConfig = {
      base: 'wan',
      key: 'a14b-lora',
      name: 'A14B LoRA',
      type: 'lora',
      variant: 'a14b',
    };

    expect(
      getGenerationValidationReasons(
        main,
        createSettings(main, { loras: [{ isEnabled: false, model: a14bLora, weight: 1 }] })
      )
    ).toEqual([]);
  });
});

describe('the guidance slider value from a model record', () => {
  // guidanceLabel selects guidance or cfg_scale so CFG-off markers cannot become guidance.
  it('prefers guidance over cfg_scale for a guidance-labelled architecture', () => {
    const model = createModel('flux', {
      variant: 'dev',
      default_settings: { cfg_scale: 1, guidance: 3.5, steps: 28 },
    } as Partial<MainModelConfig>);

    // Not 1 — that is the CFG-off marker, and buildFluxGraph wires this value into `guidance`.
    expect(getDefaultGenerateSettings(model).cfgScale).toBe(3.5);
  });

  it('falls back to cfg_scale when a guidance-labelled model records no guidance', () => {
    const model = createModel('flux', {
      variant: 'schnell',
      default_settings: { cfg_scale: 1, steps: 4 },
    } as Partial<MainModelConfig>);

    expect(getDefaultGenerateSettings(model).cfgScale).toBe(1);
  });

  it('prefers cfg_scale for a CFG-labelled architecture', () => {
    const model = createModel('sdxl', { default_settings: { cfg_scale: 6.5, steps: 30 } } as Partial<MainModelConfig>);

    expect(getDefaultGenerateSettings(model).cfgScale).toBe(6.5);
  });

  it('ignores a guidance a CFG-labelled model also records', () => {
    // Provide both defaults to expose wrong-field precedence even for non-guidance models.
    const model = createModel('sdxl', {
      default_settings: { cfg_scale: 7, guidance: 4, steps: 30 },
    } as Partial<MainModelConfig>);

    expect(getDefaultGenerateSettings(model).cfgScale).toBe(7);
  });

  it('reads the right field for every base in the table', () => {
    // Probe every architecture within bounds so clamping cannot hide wrong-field reads.
    for (const base of SUPPORTED_GENERATE_BASES) {
      const model = createModel(base, {
        default_settings: { cfg_scale: 7, guidance: 9 },
      } as Partial<MainModelConfig>);
      const label = getGenerationModelPolicy(model, createSettings(model)).ui.guidanceLabel;

      expect(getDefaultGenerateSettings(model).cfgScale, `${base} (${label}) read the wrong field`).toBe(
        label === 'Guidance' ? 9 : 7
      );
    }
  });
});

describe('the guidance range the architecture declares', () => {
  seedArchitectureCapabilities();

  /** Isolate bound errors from unrelated missing components. */
  const guidanceReasons = (model: MainModelConfig, cfgScale: number): string[] =>
    getGenerationValidationReasons(model, createSettings(model, { cfgScale })).filter((reason) =>
      /^(CFG|Guidance) must be at (least|most) /.test(reason)
    );

  it('rejects a persisted guidance above the ceiling the node enforces', () => {
    // FLUX Fill guidance 30 exceeds FLUX.2's maximum of 20.
    const model = createModel('flux2');

    expect(guidanceReasons(model, 30)).toEqual(['Guidance must be at most 20 for flux2 model.']);
    expect(guidanceReasons(model, 20)).toEqual([]);
  });

  it('leaves an architecture whose node declares no ceiling alone', () => {
    // `flux_denoise.guidance` is genuinely unbounded, and 30 is FLUX Fill's own recommendation.
    expect(guidanceReasons(createModel('flux', { variant: 'dev_fill' }), 30)).toEqual([]);
  });

  it('rejects a guidance below the floor the node enforces', () => {
    // `ernie_image_denoise.guidance_scale` is `ge=1.0`; the control used to offer 0 and 0.5.
    const model = createModel('ernie-image');

    expect(guidanceReasons(model, 0.5)).toEqual(['CFG must be at least 1 for ernie-image model.']);
    expect(guidanceReasons(model, 1)).toEqual([]);
  });

  it('clamps a model record whose stored default the architecture would reject', () => {
    // Clamp editable record defaults before reset can restore blocked values.
    const model = createModel('flux2', { default_settings: { cfg_scale: 50 } } as Partial<MainModelConfig>);

    expect(getDefaultGenerateSettings(model).cfgScale).toBe(20);
    expect(guidanceReasons(model, getDefaultGenerateSettings(model).cfgScale)).toEqual([]);

    const floored = createModel('z-image', { default_settings: { cfg_scale: 0 } } as Partial<MainModelConfig>);

    expect(getDefaultGenerateSettings(floored).cfgScale).toBe(1);
  });

  it('clamps the carried-over guidance when a model is selected, and says so', () => {
    // Model selection repairs guidance and reports the cleared field.
    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('flux', { variant: 'dev_fill' }), { cfgScale: 30 }),
      model: createModel('flux2'),
      models: [],
    });

    expect(result.settings.cfgScale).toBe(20);
    expect(result.clearedLabels).toContain('Guidance');
  });

  it('leaves a carried-over guidance the new model accepts untouched', () => {
    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sdxl'), { cfgScale: 7 }),
      model: createModel('flux2'),
      models: [],
    });

    expect(result.settings.cfgScale).toBe(7);
    expect(result.clearedLabels).not.toContain('Guidance');
  });

  it('enforces the served bound for every architecture in the table', () => {
    // Derive expectations for every base from served fixtures, independently of implementation.
    for (const row of architectureCapabilitiesFixture) {
      // Include variant rows to catch base-only lookup regressions.
      const model = createModel(row.base, { variant: row.variant ?? undefined } as Partial<MainModelConfig>);

      if (!isSupportedGenerateModel(model)) {
        continue;
      }

      const where = `${row.base}/${row.variant ?? '-'}`;
      const { guidance_max: max, guidance_min: min } = row.features;

      expect(guidanceReasons(model, min), `${where} rejected its own floor`).toEqual([]);

      if (min > 0) {
        expect(guidanceReasons(model, min - 0.5), `${where} accepted a value below its floor`).toHaveLength(1);
      }

      if (max !== null) {
        expect(guidanceReasons(model, max), `${where} rejected its own ceiling`).toEqual([]);
        expect(guidanceReasons(model, max + 0.5), `${where} accepted a value above its ceiling`).toHaveLength(1);
      }
    }
  });
});

describe('without the capability table', () => {
  /** Shared validation must gate every compile path, not only the widget. */
  it('blocks every model, including ones that are otherwise fine', () => {
    const model = createModel('flux', { variant: 'dev' });
    const settings = createSettings(model);

    // Seeded by the file-level fixture; drop it to stand in for a failed or in-flight fetch.
    resetArchitectureCapabilities();

    const reasons = getGenerationValidationReasons(model, settings);

    expect(reasons.length).toBeGreaterThan(0);
    // Missing-capability errors must identify backend data, not misreport model support.
    expect(reasons[0]).toMatch(/capabilities/i);
  });

  it('stops blocking once the table arrives', () => {
    const model = createModel('flux', { variant: 'dev' });
    const settings = createSettings(model);

    resetArchitectureCapabilities();
    expect(getGenerationValidationReasons(model, settings)[0]).toMatch(/capabilities/i);

    setArchitectureCapabilities(architectureCapabilitiesFixture);

    expect(getGenerationValidationReasons(model, settings)[0] ?? '').not.toMatch(/capabilities/i);
  });
});

describe('with a table that omits this architecture', () => {
  /** A loaded table may still lack the selected architecture; fail closed for missing rows. */
  const withoutBase = (base: string) => architectureCapabilitiesFixture.filter((row) => row.base !== base);

  it('blocks a supported base the backend did not describe', () => {
    const model = createModel('cogview4');
    const settings = createSettings(model);

    setArchitectureCapabilities(withoutBase('cogview4'));

    expect(getGenerationValidationReasons(model, settings)).toEqual([
      'The backend does not describe the cogview4 architecture, so it cannot be generated with.',
    ]);
  });

  it('leaves the architectures it did describe alone', () => {
    const model = createModel('sdxl');

    setArchitectureCapabilities(withoutBase('cogview4'));

    expect(getGenerationValidationReasons(model, createSettings(model))).toEqual([]);
  });

  it('never asks the table about an external generator, which has no architecture row', () => {
    setArchitectureCapabilities(withoutBase('cogview4'));

    expect(getGenerationValidationReasons(externalModel, createSettings(externalModel))).toEqual([]);
  });
});

describe('getGenerationDimensions and the variant it dispatches on', () => {
  it('answers from the variant row when one differs, which its parameter type now admits', () => {
    // Construct variant-specific dimensions because current fixtures otherwise hide a dropped variant.
    const schnell = architectureCapabilitiesFixture.find((row) => row.base === 'flux' && row.variant === 'schnell')!;

    setArchitectureCapabilities([
      ...architectureCapabilitiesFixture.filter((row) => row !== schnell),
      { ...schnell, defaults: { ...schnell.defaults!, height: 512, width: 512 } },
    ]);

    expect(getGenerationDimensions({ base: 'flux', type: 'main', variant: 'schnell' }).optimal).toBe(512);
    expect(getGenerationDimensions({ base: 'flux', type: 'main' }).optimal).toBe(1024);
  });
});
