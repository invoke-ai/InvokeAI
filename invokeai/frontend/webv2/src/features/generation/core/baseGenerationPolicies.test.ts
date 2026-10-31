import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';
import i18n from 'i18next';
import { describe, expect, it } from 'vitest';

import type {
  ComponentModelConfig,
  GenerateReferenceImage,
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
  MainModelConfig,
  VaeModelConfig,
} from './types';

import {
  coerceSchedulerForGraph,
  GENERATE_MODEL_SWITCH_SUBJECTS,
  getComponentSectionPolicy,
  getAutoFlux2ComponentSourceModel,
  getDefaultGenerateSettings,
  getGenerateModelSelectionResult,
  getGenerationDimensions,
  getGenerateInvocationIssues,
  getGenerationModelAvailabilityReasons,
  getGenerationUiPolicy,
  getGenerationValidationIssues,
  getGenerationValidationReasons,
  getKrea2EditLoraState,
  getMaxReferenceImages,
  getMissingReferenceImageFormat,
  getPromptPolicy,
  getSchedulerOptions,
  getSettingsWithModelDefaults,
  isReferenceImageSupported,
  isGenerateModelSelectable,
  isLossyModelSwitch,
  isSupportedGenerateModel,
  QWEN_IMAGE_21_STANDALONE_ENCODER_REASON,
  withAddedReferenceImages,
} from './baseGenerationPolicies';
import { describeGenerationIssue, getGenerationIssueFixPatch } from './generationIssues';
import { collectGenerationIssueScenarios } from './generationIssues.testing';
import { getActiveRefiner } from './sdxlRefiner';
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
const clipEmbed: ComponentModelConfig = {
  base: 'any',
  key: 'clip',
  name: 'CLIP Embed',
  type: 'clip_embed',
  variant: 'large',
};
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
const fluxLora: LoraModelConfig = { base: 'flux', key: 'flux-lora', name: 'FLUX LoRA', type: 'lora' };
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

    expect(getSchedulerOptions(createModel('sdxl')).map((option) => option.value)).toContain('euler_a');
    expect(getSchedulerOptions(flux).map((option) => option.value)).toEqual(['euler', 'heun', 'lcm']);
    expect(getSchedulerOptions(zbase).map((option) => option.value)).toEqual(['euler', 'heun']);
    expect(getSchedulerOptions(anima).map((option) => option.value)).toEqual([
      'euler',
      'heun',
      'dpmpp_2m',
      'dpmpp_2m_sde',
      'er_sde',
      'lcm',
    ]);
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
    expect(getGenerationUiPolicy(createModel('sd-1'))).toMatchObject({
      clipSkipMax: 12,
      cfgRescaleVisible: true,
      hiDiffusionVisible: true,
      schedulerVisible: true,
    });
    expect(getGenerationUiPolicy(createModel('sd-2'))).toMatchObject({
      clipSkipMax: 24,
      cfgRescaleVisible: true,
      hiDiffusionVisible: false,
    });
    expect(getGenerationUiPolicy(createModel('sdxl'))).toMatchObject({
      hiDiffusionVisible: true,
    });
    expect(getGenerationUiPolicy(createModel('flux'))).toMatchObject({
      guidanceLabel: 'Guidance',
      hiDiffusionVisible: false,
      schedulerVisible: true,
      clipSkipMax: null,
    });
    expect(getGenerationUiPolicy(createModel('sd-3'))).toMatchObject({
      schedulerVisible: false,
      sdVaeVisible: false,
    });
    expect(getGenerationUiPolicy(createModel('qwen-image')).schedulerVisible).toBe(false);
    expect(getGenerationUiPolicy(externalModel).seedVisible).toBe(false);
    expect(getGenerationUiPolicy({ ...externalModel, capabilities: { supports_seed: true } }).seedVisible).toBe(true);
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

    it('treats a GGUF branch like a safetensors one', () => {
      const ggufUnconditional = createModel('ideogram-4', { branch: 'unconditional', format: 'gguf_quantized' });
      const ggufConditional = createModel('ideogram-4', { branch: 'conditional', format: 'gguf_quantized' });
      const missing = 'Generate needs the unconditional transformer for single-file Ideogram 4 models.';

      expect(isGenerateModelSelectable(ggufUnconditional)).toBe(false);
      expect(isGenerateModelSelectable(ggufConditional)).toBe(true);
      // The slot that hides it from the picker is the slot that has to accept it.
      expect(getGenerationValidationReasons(ggufConditional, createSettings(ggufConditional))).toContain(missing);
      expect(
        getGenerationValidationReasons(
          ggufConditional,
          createSettings(ggufConditional, { ideogram4UnconditionalModel: ggufUnconditional })
        )
      ).not.toContain(missing);
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

  describe('Z-Image Qwen3 encoder', () => {
    const model = createModel('z-image', { format: 'checkpoint' });
    const qwen3_4b: ComponentModelConfig = { ...qwen3Encoder, key: 'qwen3-4b', name: 'Qwen3 4B', variant: 'qwen3_4b' };
    const qwen3_8b: ComponentModelConfig = { ...qwen3Encoder, key: 'qwen3-8b', name: 'Qwen3 8B', variant: 'qwen3_8b' };
    const source = createModel('z-image', { key: 'z-image-diffusers', format: 'diffusers' });
    const incompatible =
      'The selected Qwen3 Encoder is not the 4B one Z-Image conditions on. Select a Qwen3 4B Encoder or clear it.';

    it('requires a Qwen3 4B encoder and a FLUX VAE without a component source', () => {
      expect(getGenerationValidationReasons(model, createSettings(model))).toEqual([
        'Generate needs a Qwen3 4B Encoder for Z-Image models.',
        'Generate needs a VAE for Z-Image models.',
      ]);
      expect(
        getGenerationValidationReasons(model, createSettings(model, { qwen3EncoderModel: qwen3_4b, vae: fluxVae }))
      ).toEqual([]);
    });

    it('names a saved 8B encoder as incompatible, with or without a component source to fall back on', () => {
      expect(
        getGenerationValidationReasons(model, createSettings(model, { qwen3EncoderModel: qwen3_8b, vae: fluxVae }))
      ).toEqual([incompatible]);
      expect(
        getGenerationValidationReasons(
          model,
          createSettings(model, { componentSourceModel: source, qwen3EncoderModel: qwen3_8b })
        )
      ).toEqual([incompatible]);
      expect(getGenerationValidationReasons(model, createSettings(model, { componentSourceModel: source }))).toEqual(
        []
      );
    });

    it('offers only the 4B encoder and clears an 8B one when switching to Z-Image', () => {
      const settings = createSettings(model);
      const slot = getComponentSectionPolicy(model, settings).slots.find((s) => s.key === 'qwen3EncoderModel');
      const ctx = { model, selectedComponents: { ...settings }, settings };
      const klein9b = createModel('flux2', { format: 'gguf_quantized', variant: 'klein_9b' });

      expect(
        [qwen3_4b, qwen3_8b, animaQwen3Encoder].map((encoder) => slot?.filter?.(encoder as GenerateModelConfig, ctx))
      ).toEqual([true, false, false]);

      const result = getGenerateModelSelectionResult({
        currentValues: createSettings(klein9b, { qwen3EncoderModel: qwen3_8b, vae: fluxVae }),
        model,
        models: [],
      });

      expect(result.settings.qwen3EncoderModel).toBeNull();
      expect(result.changes.map(({ subject }) => subject)).toContain('qwen3EncoderModel');
    });
  });

  it('offers FLUX.1 only CLIP-L encoders, the one its text encoder runs', () => {
    const model = createModel('flux');
    const settings = createSettings(model);
    const clipFilter = getComponentSectionPolicy(model, settings).slots.find(
      (slot) => slot.key === 'clipEmbedModel'
    )?.filter;
    const ctx = { model, selectedComponents: { ...settings }, settings };
    const clipG = { ...clipEmbed, key: 'clip-g', variant: 'gigantic' };

    expect(clipFilter?.(clipEmbed as GenerateModelConfig, ctx)).toBe(true);
    expect(clipFilter?.(clipG as GenerateModelConfig, ctx)).toBe(false);
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
      'Width must be between 64 and 4096.'
    );
    expect(getGenerationValidationReasons(model, createSettings(model, { height: 4104 }))).toContain(
      'Height must be between 64 and 4096.'
    );
    expect(getGenerationValidationReasons(model, createSettings(model, { height: 888 }))).toContain(
      'Height must be a multiple of 16.'
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

  it('validates the external mode being compiled against the modes the model advertises', () => {
    const model: GenerateModelConfig = {
      ...externalModel,
      capabilities: { modes: ['txt2img', 'inpaint'] },
    };

    expect(getGenerationValidationReasons(model, createSettings(model))).toEqual([]);
    expect(getGenerationValidationReasons(model, createSettings(model), 'inpaint')).toEqual([]);
    expect(getGenerationValidationReasons(model, createSettings(model), 'img2img')).toContain(
      'External Model does not support image-to-image generation.'
    );
    // Modes default to txt2img only when the record omits them.
    expect(
      getGenerationValidationReasons({ ...externalModel, capabilities: null }, createSettings(externalModel), 'inpaint')
    ).toContain('External Model does not support inpainting.');
  });

  it('accepts provider sizes off the local model grids for external generators', () => {
    expect(getGenerationValidationReasons(externalModel, createSettings(externalModel, { width: 1365 }))).toEqual([]);
  });

  it('rejects dimensions above the provider maximum on either axis', () => {
    const model: GenerateModelConfig = {
      ...externalModel,
      capabilities: { modes: ['txt2img'], max_image_size: { width: 2048, height: 1536 } },
    };
    expect(getGenerationValidationReasons(model, createSettings(model, { width: 2048, height: 1536 }))).toEqual([]);
    for (const dimensions of [
      { width: 2049, height: 1536 },
      { width: 2048, height: 1537 },
    ]) {
      expect(getGenerationValidationReasons(model, createSettings(model, dimensions))).toContain(
        'External Model supports at most 2048×1536.'
      );
    }
  });

  it('counts the canvas init and output against the Seedream total image budget', () => {
    const model: GenerateModelConfig = {
      ...externalModel,
      provider_id: 'seedream',
      provider_model_id: 'seedream-4-5-251128',
      capabilities: { modes: ['txt2img', 'img2img'], supports_reference_images: true, max_reference_images: 14 },
    };
    const settings = createSettings(model, {
      referenceImages: Array.from({ length: 14 }, (_, i) => ({
        id: `reference-${i}`,
        isEnabled: true,
        config: {
          type: 'external_reference_image',
          image: {
            imageName: `reference-${i}.png`,
            width: 512,
            height: 512,
            original: { image: { image_name: `reference-${i}.png`, width: 512, height: 512 } },
          },
        },
      })),
    });
    expect(getGenerationValidationReasons(model, settings, 'txt2img')).toEqual([]);
    expect(getGenerationValidationReasons(model, settings, 'img2img')).toEqual([
      'External Model supports at most 15 images total (reference + init + output), got 16.',
    ]);
    settings.referenceImages[0]!.isEnabled = false;
    expect(getGenerationValidationReasons(model, settings, 'img2img')).toEqual([]);
    settings.referenceImages[0]!.isEnabled = true;
    // A record that does not take references sends none, so the budget counts none; the references are refused
    // on their own terms instead.
    const reasons = getGenerationValidationReasons(
      { ...model, capabilities: { modes: ['txt2img', 'img2img'], supports_reference_images: false } },
      settings,
      'img2img'
    );
    expect(reasons).toContain('External Model cannot use reference image 1. Turn it off, or remove it.');
    expect(reasons.filter((reason) => reason.includes('images total'))).toEqual([]);
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

  it('keeps concepts but clears incompatible selections when the selected model changes', () => {
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
    // The SD 1 concept stays listed and on: the SDXL graph skips it, and an SD 1 model runs it again.
    expect(result.settings.loras).toBe(settings.loras);
    expect(result.settings.vae).toEqual(sdxlVae);
    expect(result.settings.t5EncoderModel).toBeNull();
    expect(result.settings.clipSkip).toBe(0);
    expect(result.settings.cfgRescaleMultiplier).toBe(0);
    expect(result.settings.seamlessXAxis).toBe(true);
    expect(result.changes).toEqual([
      { count: 1, kind: 'turnedOff', subject: 'concepts' },
      { kind: 'removed', subject: 't5EncoderModel' },
      { kind: 'reset', subject: 'clipSkip' },
      { kind: 'reset', subject: 'cfgRescale' },
    ]);
    expect(isLossyModelSwitch(result)).toBe(true);
  });

  it('keeps concepts and their weights through SDXL -> FLUX -> SDXL, reporting only the ones that stop', () => {
    const sdxl = createModel('sdxl');
    const flux = createModel('flux');
    const loras = [
      { isEnabled: true, model: sdxlLora, weight: 0.65 },
      { isEnabled: false, model: { ...sdxlLora, key: 'sdxl-off' }, weight: 1.2 },
      { isEnabled: true, model: sd1Lora, weight: 0.3 },
    ];
    const toFlux = getGenerateModelSelectionResult({
      currentValues: createSettings(sdxl, { loras }),
      model: flux,
      models: [sdxl, flux],
    });

    // Only the SDXL concept that ran stops; the one switched off and the SD 1 one were not running anyway.
    expect(toFlux.settings.loras).toBe(loras);
    expect(toFlux.changes).toEqual([{ count: 1, kind: 'turnedOff', subject: 'concepts' }]);
    expect(isLossyModelSwitch(toFlux)).toBe(false);

    const back = getGenerateModelSelectionResult({ currentValues: toFlux.settings, model: sdxl, models: [sdxl, flux] });

    expect(back.settings.loras).toBe(loras);
    expect(back.changes).toEqual([]);
  });

  it('reports a VAE precision reset only when the user picked the precision', () => {
    const flux = createModel('flux');
    const select = (sdxl: MainModelConfig) =>
      getGenerateModelSelectionResult({
        currentValues: createSettings(sdxl, { vaePrecision: 'fp16' }),
        model: flux,
        models: [sdxl, flux],
      });
    // fp16 is this SDXL model's own default, so moving to FLUX's fp32 undoes no choice.
    const atDefault = select(createModel('sdxl', { default_settings: { vae_precision: 'fp16' } }));
    const picked = select(createModel('sdxl'));

    expect(getGenerationUiPolicy(flux).vaePrecisionVisible).toBe(false);
    expect(atDefault.settings.vaePrecision).toBe('fp32');
    expect(atDefault.changes).toEqual([]);
    expect(picked.settings.vaePrecision).toBe('fp32');
    expect(picked.changes).toEqual([{ kind: 'reset', subject: 'vaePrecision' }]);
  });

  it('initializes a provider size on model changes while preserving supported and same-model dimensions', () => {
    const model: GenerateModelConfig = {
      base: 'external',
      format: 'external_api',
      key: 'gemini',
      name: 'Gemini',
      provider_id: 'gemini',
      type: 'external_image_generator',
      capabilities: {
        modes: ['txt2img'],
        resolution_presets: [
          { aspect_ratio: '1:1', height: 1024, image_size: '1K', label: 'Square', width: 1024 },
          { aspect_ratio: '16:9', height: 1152, image_size: '2K', label: 'Landscape', width: 2048 },
        ],
      },
    };
    const select = (settings: GenerateSettings) =>
      getGenerateModelSelectionResult({ currentValues: settings, model, models: [] }).settings;

    expect(select(createSettings(createModel('sd-1'), { height: 512, width: 512 }))).toMatchObject({
      externalImageSize: '1K',
      height: 1024,
      width: 1024,
    });
    expect(select(createSettings(createModel('sdxl'), { height: 576, width: 1024 }))).toMatchObject({
      aspectRatioId: '16:9',
      aspectRatioValue: 2048 / 1152,
      externalImageSize: '2K',
      height: 1152,
      width: 2048,
    });
    expect(
      select(createSettings(createModel('sdxl'), { externalImageSize: 'stale', height: 1152, width: 2048 }))
    ).toMatchObject({
      externalImageSize: '2K',
      height: 1152,
      width: 2048,
    });
    expect(select(createSettings(model, { height: 1200, width: 1600 }))).toMatchObject({
      externalImageSize: null,
      height: 1200,
      width: 1600,
    });
    const local = getGenerateModelSelectionResult({
      currentValues: createSettings(model, { externalImageSize: '2K', height: 1152, width: 2048 }),
      model: createModel('sdxl'),
      models: [],
    }).settings;

    expect(local).toMatchObject({ externalImageSize: null, height: 1152, width: 2048 });
  });

  it('prefers the nearer pixel budget among presets of one nominal aspect ratio', () => {
    // Rounded 16:9 presets have slightly different pixel ratios (1.75 vs 1.78), so an exact ratio tie never occurs.
    const model: GenerateModelConfig = {
      base: 'external',
      format: 'external_api',
      key: 'gemini',
      name: 'Gemini',
      provider_id: 'gemini',
      type: 'external_image_generator',
      capabilities: {
        modes: ['txt2img'],
        resolution_presets: [
          { aspect_ratio: '16:9', height: 1152, image_size: '2K', label: '16:9 (2K)', width: 2048 },
          { aspect_ratio: '16:9', height: 768, image_size: '1K', label: '16:9 (1K)', width: 1344 },
          { aspect_ratio: '4:3', height: 768, image_size: '1K', label: '4:3 (1K)', width: 1024 },
        ],
      },
    };

    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sdxl'), { height: 576, width: 1024 }),
      model,
      models: [],
    });

    expect(result.settings).toMatchObject({ externalImageSize: '1K', height: 768, width: 1344 });
  });

  it('turns HiDiffusion off when changing to an unsupported model family', () => {
    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sd-1'), { hiDiffusionEnabled: true }),
      model: createModel('flux'),
      models: [],
    });

    expect(result.settings.hiDiffusionEnabled).toBe(false);
    expect(result.changes.map(({ subject }) => subject)).toContain('hiDiffusion');
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
      expect(result.changes).toEqual([]);
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
    expect(result.changes).toEqual([]);
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

  it('reconciles dimensions to the new model grid as an adjustment, not a removed setting', () => {
    const model = createModel('cogview4');
    const settings = createSettings(createModel('sdxl'), { height: 520, width: 520 });
    const result = getGenerateModelSelectionResult({ currentValues: settings, model, models: [] });

    expect(result.settings.width).toBe(512);
    expect(result.settings.height).toBe(512);
    expect(result.changes).toEqual([]);
    expect(result.dimensionsAdjusted).toBe(true);
  });

  it('reports a snap to a provider size as an adjustment, so the switch is not lossy', () => {
    const seedream: GenerateModelConfig = {
      ...externalModel,
      key: 'seedream',
      name: 'Seedream 4.5',
      provider_id: 'seedream',
      capabilities: {
        modes: ['txt2img', 'img2img'],
        aspect_ratio_sizes: {
          '1:1': { width: 2048, height: 2048 },
          '16:9': { width: 2848, height: 1600 },
        },
      },
    };
    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sdxl'), { height: 1024, width: 1024 }),
      model: seedream,
      models: [],
    });

    expect(result.settings).toMatchObject({ height: 2048, width: 2048 });
    expect(result.changes).toEqual([]);
    expect(result.dimensionsAdjusted).toBe(true);
  });

  it('keeps the Canvas frame when switching to a provider with fixed sizes', () => {
    const gptImage: GenerateModelConfig = {
      ...externalModel,
      key: 'gpt-image',
      capabilities: {
        modes: ['txt2img', 'img2img'],
        aspect_ratio_sizes: { '1:1': { width: 1024, height: 1024 }, '3:2': { width: 1536, height: 1024 } },
      },
    };
    const currentValues = createSettings(createModel('sdxl'), { height: 512, width: 640 });

    // Generate snaps to the provider's size; the Canvas sync would carry that to the frame, so a frame is kept.
    expect(getGenerateModelSelectionResult({ currentValues, model: gptImage, models: [] }).settings).toMatchObject({
      height: 1024,
      width: 1024,
    });
    const canvas = getGenerateModelSelectionResult({ currentValues, isCanvasFrame: true, model: gptImage, models: [] });
    expect(canvas.settings).toMatchObject({ height: 512, width: 640 });
    expect(canvas.dimensionsAdjusted).toBe(false);
  });

  it('reports the grid clamp from an off-grid provider size back to a local model as an adjustment', () => {
    const provider: GenerateModelConfig = { ...externalModel, key: 'gemini', provider_id: 'gemini' };
    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(provider, { height: 1024, width: 683 }),
      model: createModel('sdxl'),
      models: [],
    });

    expect(result.settings.width % 8).toBe(0);
    expect(result.settings.height).toBe(1024);
    expect(result.changes).toEqual([]);
    expect(result.dimensionsAdjusted).toBe(true);
  });

  it('reports no adjustment when the size already fits the new model', () => {
    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sd-1'), { height: 1024, width: 1024 }),
      model: createModel('sdxl'),
      models: [],
    });

    expect(result.dimensionsAdjusted).toBe(false);
  });

  it('leaves concepts as they are when applying the defaults of another family, as recall does', () => {
    const loras = [{ isEnabled: true, model: sdxlLora, weight: 0.6 }];

    expect(
      getSettingsWithModelDefaults(createSettings(createModel('sdxl'), { loras }), createModel('flux')).loras
    ).toBe(loras);
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
      // Only a concept FLUX would load must be installed; one set aside or switched off is skipped.
      loras: [
        { isEnabled: true, model: fluxLora, weight: 1 },
        { isEnabled: false, model: { ...fluxLora, key: 'flux-off' }, weight: 1 },
        { isEnabled: true, model: sd1Lora, weight: 1 },
      ],
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
    });

    expect(getGenerationModelAvailabilityReasons(model, settings, [model as never, t5Encoder as never])).toEqual([
      'CLIP Embed "CLIP Embed" is not installed. Choose another, or clear it.',
      'VAE "FLUX VAE" is not installed. Choose another, or clear it.',
      'Concept "FLUX LoRA" is not installed. Remove it, or reinstall it.',
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
    expect(result.changes).toEqual([{ kind: 'removed', subject: 'qwen3EncoderModel' }]);
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
    expect(toDev.changes.map(({ subject }) => subject)).toContain('qwen3EncoderModel');
    expect(toKlein.settings.mistralEncoderModel).toBeNull();
    expect(toKlein.changes.map(({ subject }) => subject)).toContain('mistralEncoderModel');
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
    ).toContain('Reference image 1 needs an image.');

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
    ).toContain(`Reference image 1 needs an IP Adapter model for ${model.name}.`);
  });

  it('keeps reference images, switched off, when switching to a model without reference image support', () => {
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

    expect(result.settings.referenceImages).toEqual([{ ...settings.referenceImages[0], isEnabled: false }]);
    expect(result.changes).toEqual([{ count: 1, kind: 'turnedOff', subject: 'referenceImages' }]);
    expect(isLossyModelSwitch(result)).toBe(false);
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
    expect(result.changes.map(({ subject }) => subject)).toContain('referenceImages');
  });

  it('confirms a re-target that drops tuning, but only adjusts one still at the old family defaults', () => {
    const sdxl = createModel('sdxl');
    const sd1 = createModel('sd-1');
    const ipAdapterReference = (weight: number) => ({
      config: {
        beginEndStepPct: [0, 1] as [number, number],
        clipVisionModel: 'ViT-H' as const,
        image: referenceImage,
        method: 'full' as const,
        model: sdxlIpAdapter,
        type: 'ip_adapter' as const,
        weight,
      },
      id: 'ref-1',
      isEnabled: true,
    });
    const select = (weight: number) =>
      getGenerateModelSelectionResult({
        currentValues: createSettings(sdxl, { referenceImages: [ipAdapterReference(weight)] }),
        model: sd1,
        models: [sdxl, sd1IpAdapter, sdxlIpAdapter],
      });

    expect(select(1).changes).toEqual([{ count: 1, kind: 'adjusted', subject: 'referenceImages' }]);
    expect(isLossyModelSwitch(select(1))).toBe(false);
    expect(select(0.4).changes).toEqual([{ count: 1, kind: 'reset', subject: 'referenceImages' }]);
    expect(isLossyModelSwitch(select(0.4))).toBe(true);
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
    expect(result.changes.map(({ subject }) => subject)).not.toContain('referenceImages');
  });

  it('starts a Krea-2 reference list with a style reference and re-targets other kinds to it', () => {
    // A Diffusers bundle needs no standalone components, which keeps the reasons to the reference check.
    const krea2 = createModel('krea-2', { format: 'diffusers' });

    expect(isReferenceImageSupported(krea2)).toBe(true);
    // One style reference, or up to three edit references.
    expect(getMaxReferenceImages(krea2)).toBe(3);
    expect(
      withAddedReferenceImages(krea2, [sdxlIpAdapter], [], [{ id: 'new', image: referenceImage }])[0]?.config
    ).toEqual({ image: referenceImage, styleStrength: 1, type: 'krea2_reference_image' });

    const fromSdxl = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sdxl'), {
        referenceImages: [
          {
            id: 'ref-1',
            isEnabled: true,
            config: {
              beginEndStepPct: [0, 1],
              clipVisionModel: 'ViT-H',
              image: referenceImage,
              method: 'style',
              model: sdxlIpAdapter,
              type: 'ip_adapter',
              weight: 1,
            },
          },
        ],
      }),
      model: krea2,
      models: [sdxlIpAdapter],
    });

    expect(fromSdxl.settings.referenceImages).toEqual([
      {
        id: 'ref-1',
        isEnabled: true,
        config: { image: referenceImage, styleStrength: 1, type: 'krea2_reference_image' },
      },
    ]);

    const styleSettings = createSettings(krea2, {
      referenceImages: [
        {
          id: 'style',
          isEnabled: true,
          config: { image: referenceImage, styleStrength: 1, type: 'krea2_reference_image' },
        },
      ],
    });

    expect(getGenerationValidationReasons(krea2, styleSettings)).toEqual([]);
    expect(getGenerationValidationReasons(createModel('sd-3'), styleSettings)).toEqual([
      'sd-3 model cannot use reference image 1. Turn it off, or remove it.',
    ]);
  });

  it('switches off the enabled references a newly selected model has no room for, keeping every reference', () => {
    const singleReferenceModel: GenerateModelConfig = {
      ...externalModel,
      capabilities: { ...externalModel.capabilities, max_reference_images: 1, supports_reference_images: true },
    };
    const ipAdapter = (id: string, isEnabled: boolean) => ({
      id,
      isEnabled,
      config: {
        beginEndStepPct: [0, 1] as [number, number],
        clipVisionModel: 'ViT-H' as const,
        image: {
          ...referenceImage,
          original: { image: { ...referenceImage.original.image, image_name: `${id}.png` } },
        },
        method: 'style' as const,
        model: sdxlIpAdapter,
        type: 'ip_adapter' as const,
        weight: 1,
      },
    });

    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sdxl'), {
        referenceImages: [ipAdapter('bypassed', false), ipAdapter('used', true), ipAdapter('extra', true)],
      }),
      model: singleReferenceModel,
      models: [sdxlIpAdapter],
    });

    // The first enabled reference is the one that keeps running, not the first in the list.
    expect(
      result.settings.referenceImages.map(({ config, id, isEnabled }) => ({ id, isEnabled, type: config.type }))
    ).toEqual([
      { id: 'bypassed', isEnabled: false, type: 'external_reference_image' },
      { id: 'used', isEnabled: true, type: 'external_reference_image' },
      { id: 'extra', isEnabled: false, type: 'external_reference_image' },
    ]);
    expect(result.changes.map(({ subject }) => subject)).toContain('referenceImages');
    expect(
      getGenerationValidationReasons(singleReferenceModel, result.settings).filter((reason) =>
        reason.includes('reference')
      )
    ).toEqual([]);
  });

  it('names a limit of one reference in the singular', () => {
    const singleReferenceModel: GenerateModelConfig = {
      ...externalModel,
      capabilities: { ...externalModel.capabilities, max_reference_images: 1, supports_reference_images: true },
    };
    const external = (id: string) => ({
      id,
      isEnabled: true,
      config: { image: referenceImage, type: 'external_reference_image' as const },
    });

    expect(
      getGenerationValidationReasons(
        singleReferenceModel,
        createSettings(singleReferenceModel, { referenceImages: [external('a'), external('b')] })
      )
    ).toContain(`${singleReferenceModel.name} supports at most 1 reference image. Turn off or remove the extra ones.`);
  });

  describe('Krea-2 style and edit references', () => {
    const krea2 = createModel('krea-2', { format: 'diffusers' });
    const style = (id: string, isEnabled = true) => ({
      id,
      isEnabled,
      config: { image: referenceImage, styleStrength: 1, type: 'krea2_reference_image' as const },
    });
    const edit = (id: string, isEnabled = true) => ({
      id,
      isEnabled,
      config: { image: referenceImage, type: 'krea2_edit_reference_image' as const },
    });
    const reasonsFor = (model: MainModelConfig, overrides: Partial<GenerateSettings>) =>
      getGenerationValidationReasons(model, createSettings(model, overrides));
    const ggufEncoder: ComponentModelConfig = {
      base: 'any',
      format: 'gguf_quantized',
      key: 'qwen3-vl-gguf',
      name: 'Qwen3-VL 4B Q4_K_M',
      type: 'qwen3_vl_encoder',
      variant: 'qwen3_vl_4b',
    };
    const ggufReason = `Krea-2 edit references need a Qwen3-VL encoder with vision weights, which the GGUF encoder "${ggufEncoder.name}" does not have. Select a safetensors file or Diffusers folder that includes the vision weights.`;

    it('types a new Krea-2 reference as style only when it is the one running reference', () => {
      const typesAfterAdding = (count: number, referenceImages: GenerateSettings['referenceImages'] = []) =>
        withAddedReferenceImages(
          krea2,
          [],
          referenceImages,
          Array.from({ length: count }, (_, index) => ({ id: `new-${index}`, image: referenceImage }))
        ).map(({ config, id, isEnabled }) => [id, config.type, isEnabled]);

      expect(typesAfterAdding(1)).toEqual([['new-0', 'krea2_reference_image', true]]);
      // Several images at once are an edit set, never a pile of style references.
      expect(typesAfterAdding(3).map(([, type]) => type)).toEqual([
        'krea2_edit_reference_image',
        'krea2_edit_reference_image',
        'krea2_edit_reference_image',
      ]);
      expect(typesAfterAdding(1, [edit('a')])).toEqual([
        ['a', 'krea2_edit_reference_image', true],
        ['new-0', 'krea2_edit_reference_image', true],
      ]);
      // Switched-off and zero-strength references do not run, so they leave the newcomer alone.
      const bypassed = { ...style('s'), config: { ...style('s').config, styleStrength: 0 } };

      expect(typesAfterAdding(1, [edit('a', false), bypassed])).toEqual([
        ['a', 'krea2_edit_reference_image', false],
        ['s', 'krea2_reference_image', true],
        ['new-0', 'krea2_reference_image', true],
      ]);
    });

    it('turns a lone untouched style reference into an edit reference when another image is added', () => {
      const added = (referenceImages: GenerateSettings['referenceImages']) =>
        withAddedReferenceImages(krea2, [], referenceImages, [{ id: 'new', image: referenceImage }]);

      // Building an edit set one image at a time: the first card started as Style at its default strength.
      expect(added([style('s')])).toEqual([edit('s'), edit('new')]);
      expect(reasonsFor(krea2, { referenceImages: added([style('s')]) })).toEqual([]);
      // Picking three images one by one, as the gallery picker hands them over, ends as an edit set.
      const picked = ['a', 'b', 'c'].reduce<GenerateSettings['referenceImages']>(
        (list, id) => withAddedReferenceImages(krea2, [], list, [{ id, image: referenceImage }]),
        []
      );

      expect(picked).toEqual([edit('a'), edit('b'), edit('c')]);

      // A changed strength marks the card as meant for style: it stays, and the mix is reported, not hidden.
      const tuned = { ...style('s'), config: { ...style('s').config, styleStrength: 1.4 } };

      expect(added([tuned])).toEqual([tuned, edit('new')]);
      expect(reasonsFor(krea2, { referenceImages: added([tuned]) })).toEqual([
        'Krea-2 cannot combine a style reference with edit references. Switch the style reference to Edit, or turn off one kind.',
      ]);
      // Only a lone running style reference flips; a switched-off one stays as it is.
      expect(added([style('off', false), edit('a')])).toEqual([style('off', false), edit('a'), edit('new')]);
      // Other families have one kind, so nothing changes mode.
      const flux2 = createModel('flux2', { format: 'diffusers' });
      const flux2Reference = {
        config: { image: referenceImage, type: 'flux2_reference_image' as const },
        id: 'f',
        isEnabled: true,
      };

      expect(withAddedReferenceImages(flux2, [], [flux2Reference], [{ id: 'new', image: referenceImage }])).toEqual([
        flux2Reference,
        { ...flux2Reference, id: 'new' },
      ]);
    });

    it('treats a zero-strength style reference as off in every check', () => {
      const bypassed = { ...style('bypassed'), config: { ...style('bypassed').config, styleStrength: 0 } };

      expect(reasonsFor(krea2, { referenceImages: [bypassed, style('a')] })).toEqual([]);
      expect(reasonsFor(krea2, { referenceImages: [bypassed, edit('a'), edit('b'), edit('c')] })).toEqual([]);

      // It does not take precedence over the edit set when the model changes.
      const otherKrea2 = createModel('krea-2', { format: 'diffusers', key: 'krea2-other' });
      const switched = getGenerateModelSelectionResult({
        currentValues: createSettings(krea2, { referenceImages: [bypassed, edit('a'), edit('b')] }),
        model: otherKrea2,
        models: [],
      }).settings.referenceImages;

      expect(switched.map(({ id, isEnabled }) => [id, isEnabled])).toEqual([
        ['bypassed', true],
        ['a', true],
        ['b', true],
      ]);
    });

    it('accepts up to three edit references and counts only enabled ones', () => {
      expect(reasonsFor(krea2, { referenceImages: [edit('a'), edit('b'), edit('c')] })).toEqual([]);
      expect(reasonsFor(krea2, { referenceImages: [edit('a'), edit('b'), edit('c'), edit('d', false)] })).toEqual([]);
      expect(reasonsFor(krea2, { referenceImages: [edit('a'), edit('b'), edit('c'), edit('d')] })).toContain(
        `${krea2.name} supports at most 3 reference images. Turn off or remove the extra ones.`
      );
    });

    it('runs one enabled style reference at a time and never mixes style with edit references', () => {
      const styleReason =
        'Krea-2 uses one style reference at a time. Turn off all but one, or switch all of them to Edit.';
      const mixReason =
        'Krea-2 cannot combine a style reference with edit references. Switch the style reference to Edit, or turn off one kind.';

      expect(reasonsFor(krea2, { referenceImages: [style('a'), style('b')] })).toEqual([styleReason]);
      expect(reasonsFor(krea2, { referenceImages: [style('a'), edit('b')] })).toEqual([mixReason]);
      // A switched-off reference of the other kind stays out of the check.
      expect(reasonsFor(krea2, { referenceImages: [style('a', false), style('b'), edit('c', false)] })).toEqual([]);
      expect(reasonsFor(krea2, { referenceImages: [style('a', false), edit('b'), edit('c')] })).toEqual([]);
    });

    it('says whether running edit references have an enabled Krea-2 LoRA to read them', () => {
      const krea2Lora: LoraModelConfig = { base: 'krea-2', key: 'pose-lora', name: 'Pose LoRA', type: 'lora' };
      const lora = (model: LoraModelConfig, isEnabled = true) => ({ isEnabled, model, weight: 1 });
      const state = (overrides: Partial<GenerateSettings>, model: MainModelConfig = krea2) =>
        getKrea2EditLoraState(model, createSettings(model, overrides));

      expect(state({ loras: [lora(krea2Lora)], referenceImages: [edit('a')] })).toBe('ready');
      // A disabled LoRA or one for another family does not reach the graph.
      expect(state({ loras: [lora(krea2Lora, false), lora(sdxlLora)], referenceImages: [edit('a')] })).toBe('noLora');
      expect(state({ loras: [lora(krea2Lora)], referenceImages: [edit('a', false), style('s')] })).toBe(
        'noEditReferences'
      );
      expect(state({ loras: [lora(krea2Lora)], referenceImages: [edit('a')] }, createModel('flux2'))).toBe(
        'noEditReferences'
      );
    });

    it('blocks edit references on a GGUF encoder, which has no vision tower, and only then', () => {
      const checkpoint = createModel('krea-2', { format: 'checkpoint', key: 'krea2-checkpoint' });
      const vae: VaeModelConfig = { base: 'qwen-image', key: 'krea2-vae', name: 'Qwen VAE', type: 'vae' };
      const safetensorsEncoder: ComponentModelConfig = { ...ggufEncoder, format: 'checkpoint', key: 'qwen3-vl-st' };

      expect(reasonsFor(checkpoint, { qwen3VLEncoderModel: ggufEncoder, referenceImages: [edit('a')], vae })).toEqual([
        ggufReason,
      ]);
      // A Diffusers main reads with the GGUF override too, when one is selected.
      expect(reasonsFor(krea2, { qwen3VLEncoderModel: ggufEncoder, referenceImages: [edit('a')] })).toEqual([
        ggufReason,
      ]);
      expect(reasonsFor(krea2, { referenceImages: [edit('a')] })).toEqual([]);
      expect(
        reasonsFor(checkpoint, { qwen3VLEncoderModel: safetensorsEncoder, referenceImages: [edit('a')], vae })
      ).toEqual([]);
      // Style transfer and switched-off edit references never reach the vision tower.
      expect(
        reasonsFor(checkpoint, {
          qwen3VLEncoderModel: ggufEncoder,
          referenceImages: [style('a'), edit('b', false)],
          vae,
        })
      ).toEqual([]);
    });

    it('re-targets references into Krea-2 as one set and switches off the ones that cannot run together', () => {
      const flux2 = createModel('flux2', { format: 'diffusers' });
      const flux2Reference = (id: string, isEnabled = true) => ({
        id,
        isEnabled,
        config: { image: referenceImage, type: 'flux2_reference_image' as const },
      });
      const select = (
        model: MainModelConfig,
        current: MainModelConfig,
        referenceImages: GenerateSettings['referenceImages']
      ) =>
        getGenerateModelSelectionResult({
          currentValues: createSettings(current, { referenceImages }),
          model,
          models: [],
        }).settings.referenceImages;

      // An edit set survives a trip through another family and back.
      const viaFlux2 = select(flux2, krea2, [edit('a'), edit('b'), edit('c')]);
      const backToKrea2 = select(krea2, flux2, viaFlux2);

      expect(backToKrea2.map(({ config, id, isEnabled }) => ({ id, isEnabled, type: config.type }))).toEqual([
        { id: 'a', isEnabled: true, type: 'krea2_edit_reference_image' },
        { id: 'b', isEnabled: true, type: 'krea2_edit_reference_image' },
        { id: 'c', isEnabled: true, type: 'krea2_edit_reference_image' },
      ]);
      expect(getGenerationValidationReasons(krea2, createSettings(krea2, { referenceImages: backToKrea2 }))).toEqual(
        []
      );

      // The one running reference becomes the style reference, and a switched-off one would join it as an edit
      // reference. Past the limit of three, the extra edit reference stops.
      expect(select(krea2, flux2, [flux2Reference('a')]).map(({ config }) => config.type)).toEqual([
        'krea2_reference_image',
      ]);
      expect(
        select(krea2, flux2, [flux2Reference('a'), flux2Reference('b', false)]).map(({ config }) => config.type)
      ).toEqual(['krea2_reference_image', 'krea2_edit_reference_image']);
      expect(
        select(
          krea2,
          flux2,
          ['a', 'b', 'c', 'd'].map((id) => flux2Reference(id))
        ).map(({ id, isEnabled }) => [id, isEnabled])
      ).toEqual([
        ['a', true],
        ['b', true],
        ['c', true],
        ['d', false],
      ]);

      // An edit set keeps running on another Krea-2 model; a mixed list keeps its first kind.
      const editSet = [edit('a'), edit('b'), edit('c')];
      const otherKrea2 = createModel('krea-2', { format: 'diffusers', key: 'krea2-other' });

      expect(
        getGenerateModelSelectionResult({
          currentValues: createSettings(krea2, { referenceImages: editSet }),
          model: otherKrea2,
          models: [],
        }).settings.referenceImages
      ).toBe(editSet);
      expect(
        getGenerateModelSelectionResult({
          currentValues: createSettings(krea2, { referenceImages: [edit('a'), style('b'), edit('c')] }),
          model: otherKrea2,
          models: [],
        }).settings.referenceImages.map(({ id, isEnabled }) => [id, isEnabled])
      ).toEqual([
        ['a', true],
        ['b', false],
        ['c', true],
      ]);

      // An edit reference after the style reference stops too, and the switch names what it changed.
      const styleFirst = getGenerateModelSelectionResult({
        currentValues: createSettings(krea2, { referenceImages: [style('s'), edit('e')] }),
        model: otherKrea2,
        models: [],
      });

      expect(styleFirst.settings.referenceImages.map(({ id, isEnabled }) => [id, isEnabled])).toEqual([
        ['s', true],
        ['e', false],
      ]);
      expect(styleFirst.changes.map(({ subject }) => subject)).toContain('referenceImages');

      // Leaving Krea-2 re-targets both kinds, keeping their images and enabled state.
      const toFlux2 = getGenerateModelSelectionResult({
        currentValues: createSettings(krea2, { referenceImages: [edit('a'), style('b', false)] }),
        model: flux2,
        models: [],
      });

      expect(toFlux2.settings.referenceImages).toEqual([flux2Reference('a'), flux2Reference('b', false)]);
    });
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

  it('offers Qwen-Image-2.1 reference images only on its Diffusers pipeline, whose encoder reads images', () => {
    const diffusers = createModel('qwen-image-2-1', { format: 'diffusers' });

    expect(isReferenceImageSupported(diffusers)).toBe(true);
    expect(getMaxReferenceImages(diffusers)).toBe(4);
    expect(isReferenceImageSupported(createModel('qwen-image-2-1', { format: 'gguf_quantized' }))).toBe(false);
    expect(isReferenceImageSupported(createModel('qwen-image-2-1', { format: 'checkpoint' }))).toBe(false);
  });

  // The UI names the missing format only when switching to it would enable references.
  it('names the format a model lacks for reference images only when the format alone blocks them', () => {
    expect(getMissingReferenceImageFormat(createModel('qwen-image-2-1', { format: 'gguf_quantized' }))).toBe(
      'diffusers'
    );
    expect(getMissingReferenceImageFormat(createModel('qwen-image-2-1', { format: 'checkpoint' }))).toBe('diffusers');
    expect(getMissingReferenceImageFormat(createModel('qwen-image-2-1', { format: 'diffusers' }))).toBeNull();
    // Qwen-Image needs the Edit variant, not a format; CogView4 takes no references at all.
    expect(getMissingReferenceImageFormat(createModel('qwen-image', { format: 'gguf_quantized' }))).toBeNull();
    expect(getMissingReferenceImageFormat(createModel('cogview4', { format: 'gguf_quantized' }))).toBeNull();
    expect(getMissingReferenceImageFormat(externalModel)).toBeNull();
    expect(getMissingReferenceImageFormat(undefined)).toBeNull();
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
      ).filter((reason) => reason.includes('supports at most'));

    expect(countReasons(2)).toEqual([]);
    expect(countReasons(3)).toEqual([
      'sdxl model supports at most 2 reference images. Turn off or remove the extra ones.',
    ]);
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
    ).toContain('sd-3 model cannot use reference image 1. Turn it off, or remove it.');
  });

  it('blocks Qwen-Image-2.1 references while a standalone encoder replaces the one that reads them', () => {
    const model = createModel('qwen-image-2-1', { format: 'diffusers' });
    const encoder = {
      base: 'any',
      key: 'qwen3-vl-8b',
      name: 'Qwen3-VL 8B',
      type: 'qwen3_vl_encoder',
      variant: 'qwen3_vl_8b',
    } as const;
    const referenceImages = [
      {
        config: { image: referenceImage, type: 'qwen_image_2_1_reference_image' as const },
        id: 'ref-1',
        isEnabled: true,
      },
    ];

    expect(
      getGenerationValidationReasons(model, createSettings(model, { qwen3VLEncoderModel: encoder, referenceImages }))
    ).toContain(QWEN_IMAGE_21_STANDALONE_ENCODER_REASON);
    expect(getGenerationValidationReasons(model, createSettings(model, { referenceImages }))).not.toContain(
      QWEN_IMAGE_21_STANDALONE_ENCODER_REASON
    );
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
    ).toContain(
      'Reference image 1 uses "SD 1 IP Adapter", which is not installed. Turn it off, or choose another model for it.'
    );
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

describe('Anima inpaint adapter slot', () => {
  const anima = createModel('anima');
  const inpaintAdapter: ComponentModelConfig = {
    base: 'anima',
    cond_in_channels: 4,
    key: 'anima-inpaint',
    name: 'Anima LLLite Inpainting',
    type: 'controlnet',
  };

  it('is offered to Anima only, optional, and takes 4-channel or pre-field Anima LLLite models', () => {
    const settings = createSettings(anima);
    const slot = getComponentSectionPolicy(anima, settings).slots.find((s) => s.key === 'animaInpaintAdapterModel');
    const ctx = { model: anima, selectedComponents: settings, settings };

    expect(slot?.modelTypes).toEqual(['controlnet']);
    expect(slot?.required).toBeUndefined();
    expect(slot?.filter?.(inpaintAdapter, ctx)).toBe(true);
    expect(slot?.filter?.({ ...inpaintAdapter, cond_in_channels: null }, ctx)).toBe(true);
    expect(slot?.filter?.({ ...inpaintAdapter, cond_in_channels: 3 }, ctx)).toBe(false);
    expect(slot?.filter?.({ ...inpaintAdapter, base: 'sdxl' }, ctx)).toBe(false);
    expect(getGenerationValidationReasons(anima, settings)).not.toContain(expect.stringContaining('Inpaint Adapter'));
    expect(
      getComponentSectionPolicy(createModel('sdxl'), settings).slots.some((s) => s.key === 'animaInpaintAdapterModel')
    ).toBe(false);
  });

  it('is cleared, and named, when the model leaves Anima', () => {
    const settings = createSettings(anima, {
      animaInpaintAdapterModel: inpaintAdapter,
      animaInpaintAdapterWeight: 0.6,
      qwen3EncoderModel: animaQwen3Encoder,
      vae: qwenImageVae,
    });

    const result = getGenerateModelSelectionResult({ currentValues: settings, model: createModel('sdxl'), models: [] });

    expect(result.settings.animaInpaintAdapterModel).toBeNull();
    expect(result.changes.map(({ subject }) => subject)).toContain('animaInpaintAdapterModel');
    expect(result.settings.animaInpaintAdapterWeight).toBe(0.6);
    expect(
      getGenerateModelSelectionResult({ currentValues: settings, model: anima, models: [] }).settings
        .animaInpaintAdapterModel
    ).toBe(inpaintAdapter);
  });
});

describe('the Qwen3.5 encoder slots of Anima-3.8B and MOSS-Image', () => {
  const qwen35Encoder = (variant: string): ComponentModelConfig => ({
    base: 'any',
    key: `qwen35-${variant}`,
    name: `Qwen3.5 ${variant}`,
    type: 'qwen3_5_encoder',
    variant,
  });
  const anima38 = createModel('anima', { format: 'checkpoint', variant: 'anima_qwen35' });
  const mossSingleFile = createModel('moss-image', { format: 'checkpoint' });
  const offers = (model: MainModelConfig, candidate: ComponentModelConfig): boolean | undefined => {
    const settings = createSettings(model);
    const slot = getComponentSectionPolicy(model, settings).slots.find((s) => s.key === 'qwen35EncoderModel');

    return slot?.filter?.(candidate, { model, selectedComponents: settings, settings });
  };

  it('offers each model only the encoder size it was trained on, though both share the setting', () => {
    expect(offers(anima38, qwen35Encoder('qwen3_5_4b'))).toBe(true);
    expect(offers(anima38, qwen35Encoder('qwen3_5_9b'))).toBe(false);
    expect(offers(mossSingleFile, qwen35Encoder('qwen3_5_9b'))).toBe(true);
    expect(offers(mossSingleFile, qwen35Encoder('qwen3_5_4b'))).toBe(false);
  });

  it('requires the encoder and VAE for a single-file MOSS-Image, and nothing for a Diffusers one', () => {
    expect(getGenerationValidationReasons(mossSingleFile, createSettings(mossSingleFile))).toEqual([
      'Generate needs a Qwen3.5 9B Encoder for non-Diffusers MOSS-Image models.',
      'Generate needs a VAE for non-Diffusers MOSS-Image models.',
    ]);
    expect(
      getGenerationValidationReasons(
        mossSingleFile,
        createSettings(mossSingleFile, { qwen35EncoderModel: qwen35Encoder('qwen3_5_9b'), vae: flux2Vae })
      )
    ).toEqual([]);

    const mossDiffusers = createModel('moss-image', { format: 'diffusers' });

    expect(getComponentSectionPolicy(mossDiffusers, createSettings(mossDiffusers)).defaultOpen).toBe(false);
    expect(getGenerationValidationReasons(mossDiffusers, createSettings(mossDiffusers))).toEqual([]);
  });

  it('clears an Anima-3.8B encoder, and names it, when the model changes to MOSS-Image', () => {
    const settings = createSettings(anima38, {
      qwen35EncoderModel: qwen35Encoder('qwen3_5_4b'),
      qwen3EncoderModel: animaQwen3Encoder,
      vae: qwenImageVae,
    });

    const result = getGenerateModelSelectionResult({ currentValues: settings, model: mossSingleFile, models: [] });

    expect(result.settings.qwen35EncoderModel).toBeNull();
    expect(result.changes.map(({ subject }) => subject)).toContain('qwen35EncoderModel');
  });
});

describe('Anima IP-Adapter references', () => {
  const animaQwen3 = createModel('anima', { format: 'checkpoint', variant: 'anima_qwen3' });
  const animaQwen35 = createModel('anima', { format: 'checkpoint', key: 'anima-3.8b', variant: 'anima_qwen35' });
  const animaIpAdapter: ComponentModelConfig = {
    base: 'anima',
    key: 'anima-ip-adapter',
    name: 'Anima IP-Adapter',
    type: 'ip_adapter',
  };
  const secondAnimaIpAdapter: ComponentModelConfig = { ...animaIpAdapter, key: 'anima-ip-adapter-2', name: 'Second' };
  const animaReference = (id: string) => ({
    config: {
      beginEndStepPct: [0, 1] as [number, number],
      clipVisionModel: 'ViT-H' as const,
      image: referenceImage,
      method: 'full' as const,
      model: animaIpAdapter,
      type: 'ip_adapter' as const,
      weight: 1,
    },
    id,
    isEnabled: true,
  });

  it('offers four references on the Qwen3 releases and none on Anima-3.8B', () => {
    expect(isReferenceImageSupported(animaQwen3)).toBe(true);
    expect(getMaxReferenceImages(animaQwen3)).toBe(4);
    expect(isReferenceImageSupported(animaQwen35)).toBe(false);
    expect(getMaxReferenceImages(animaQwen35)).toBe(0);
  });

  it('accepts four references sharing one Anima IP-Adapter and rejects a fifth', () => {
    const reasons = (count: number) =>
      getGenerationValidationReasons(
        animaQwen3,
        createSettings(animaQwen3, {
          referenceImages: Array.from({ length: count }, (_, index) => animaReference(`ref-${index}`)),
        })
      ).filter((reason) => reason.includes('reference') || reason.includes('Reference'));

    expect(reasons(4)).toEqual([]);
    expect(reasons(5)).toEqual(['anima model supports at most 4 reference images. Turn off or remove the extra ones.']);
  });

  it('rejects a strength outside the 0-2 range the Anima node accepts', () => {
    const withWeight = (weight: number) =>
      getGenerationValidationReasons(
        animaQwen3,
        createSettings(animaQwen3, {
          referenceImages: [{ ...animaReference('ref-1'), config: { ...animaReference('ref-1').config, weight } }],
        })
      ).filter((reason) => reason.includes('strength'));

    expect(withWeight(0)).toEqual([]);
    expect(withWeight(2)).toEqual([]);
    expect(withWeight(-0.5)).toEqual(['Reference image 1 strength must be between 0 and 2.']);
    expect(withWeight(2.5)).toEqual(['Reference image 1 strength must be between 0 and 2.']);
  });

  it('defaults a re-targeted reference to the first installed Anima IP-Adapter', () => {
    const sdxlModel = createModel('sdxl');
    const settings = createSettings(sdxlModel, {
      referenceImages: [
        { ...animaReference('ref-1'), config: { ...animaReference('ref-1').config, model: sdxlIpAdapter } },
      ],
    });

    const result = getGenerateModelSelectionResult({
      currentValues: settings,
      model: animaQwen3,
      models: [sdxlIpAdapter, animaIpAdapter, secondAnimaIpAdapter],
    });

    expect(result.settings.referenceImages).toEqual([
      {
        config: {
          beginEndStepPct: [0, 1],
          clipVisionModel: 'ViT-H',
          image: referenceImage,
          method: 'full',
          model: animaIpAdapter,
          type: 'ip_adapter',
          weight: 1,
        },
        id: 'ref-1',
        isEnabled: true,
      },
    ]);
  });

  it('re-targets an SD reference that never had an adapter, keeping its step window and taking the default strength', () => {
    const sdxlModel = createModel('sdxl');
    const sdxlReference = {
      config: {
        beginEndStepPct: [0.2, 0.8] as [number, number],
        clipVisionModel: 'ViT-G' as const,
        image: referenceImage,
        method: 'style' as const,
        model: null,
        type: 'ip_adapter' as const,
        weight: -0.5,
      },
      id: 'ref-1',
      isEnabled: false,
    };
    const switchTo = (models: ComponentModelConfig[]) =>
      getGenerateModelSelectionResult({
        currentValues: createSettings(sdxlModel, { referenceImages: [sdxlReference] }),
        model: animaQwen3,
        models,
      }).settings;
    const animaDefault = (model: ComponentModelConfig | null) => [
      {
        config: {
          beginEndStepPct: [0.2, 0.8],
          clipVisionModel: 'ViT-H',
          image: referenceImage,
          method: 'full',
          model,
          type: 'ip_adapter',
          weight: 1,
        },
        id: 'ref-1',
        isEnabled: false,
      },
    ];

    const withAdapter = switchTo([sdxlIpAdapter, animaIpAdapter]);
    expect(withAdapter.referenceImages).toEqual(animaDefault(animaIpAdapter));
    expect(
      getGenerationValidationReasons(animaQwen3, {
        ...withAdapter,
        referenceImages: withAdapter.referenceImages.map((ref) => ({ ...ref, isEnabled: true })),
      }).filter((reason) => reason.startsWith('Reference Image'))
    ).toEqual([]);

    // Nothing to select: the weight is still brought into range, and a second switch leaves the result alone.
    const withoutAdapter = switchTo([sdxlIpAdapter]);
    expect(withoutAdapter.referenceImages).toEqual(animaDefault(null));
    expect(
      getGenerateModelSelectionResult({ currentValues: withoutAdapter, model: animaQwen3, models: [sdxlIpAdapter] })
        .settings.referenceImages
    ).toBe(withoutAdapter.referenceImages);
  });

  it('fills in the adapter of an Anima reference once one is installed, keeping its strength and step window', () => {
    const settings = createSettings(animaQwen3, {
      referenceImages: [
        {
          ...animaReference('ref-1'),
          config: { ...animaReference('ref-1').config, beginEndStepPct: [0.3, 0.7], model: null, weight: 1.5 },
        },
      ],
    });

    const result = getGenerateModelSelectionResult({
      currentValues: settings,
      model: animaQwen3,
      models: [sdxlIpAdapter, animaIpAdapter, secondAnimaIpAdapter],
    });

    expect(result.settings.referenceImages).toEqual([
      {
        config: {
          beginEndStepPct: [0.3, 0.7],
          clipVisionModel: 'ViT-H',
          image: referenceImage,
          method: 'full',
          model: animaIpAdapter,
          type: 'ip_adapter',
          weight: 1.5,
        },
        id: 'ref-1',
        isEnabled: true,
      },
    ]);
  });

  it('keeps the chosen Anima adapter while clamping an out-of-range strength to 2', () => {
    const settings = createSettings(animaQwen3, {
      referenceImages: [
        {
          ...animaReference('ref-1'),
          config: { ...animaReference('ref-1').config, model: secondAnimaIpAdapter, weight: 3 },
        },
      ],
    });

    const [reference] = getGenerateModelSelectionResult({
      currentValues: settings,
      model: animaQwen3,
      models: [animaIpAdapter, secondAnimaIpAdapter],
    }).settings.referenceImages;

    expect(reference?.config).toMatchObject({ model: secondAnimaIpAdapter, weight: 2 });
  });

  it('keeps an SD reference without an adapter as it is on SDXL', () => {
    const sdxlModel = createModel('sdxl');
    const referenceImages = [
      {
        config: {
          beginEndStepPct: [0.2, 0.8] as [number, number],
          clipVisionModel: 'ViT-H' as const,
          image: referenceImage,
          method: 'style' as const,
          model: null,
          type: 'ip_adapter' as const,
          weight: -0.5,
        },
        id: 'ref-1',
        isEnabled: true,
      },
    ];

    expect(
      getGenerateModelSelectionResult({
        currentValues: createSettings(sdxlModel, { referenceImages }),
        model: createModel('sdxl', { key: 'sdxl-other' }),
        models: [sdxlIpAdapter],
      }).settings.referenceImages
    ).toBe(referenceImages);
  });

  it('keeps the reference, switched off, when switching to Anima-3.8B', () => {
    const reference = animaReference('ref-1');
    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(animaQwen3, { referenceImages: [reference] }),
      model: animaQwen35,
      models: [animaIpAdapter],
    });

    expect(result.settings.referenceImages).toEqual([{ ...reference, isEnabled: false }]);
    expect(result.changes).toEqual([{ count: 1, kind: 'turnedOff', subject: 'referenceImages' }]);
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

    expect(reasons).toContain('Width must be a multiple of 16.');
    expect(reasons).toContain('Height must be a multiple of 16.');
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

  it('does not block on a Wan LoRA for the other family, which the graph skips and the panel badges', () => {
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

    expect(reasons).toEqual([]);
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
      const label = getGenerationUiPolicy(model).guidanceLabel;

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
    expect(result.changes.map(({ subject }) => subject)).toContain('guidance');
  });

  it('leaves a carried-over guidance the new model accepts untouched', () => {
    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sdxl'), { cfgScale: 7 }),
      model: createModel('flux2'),
      models: [],
    });

    expect(result.settings.cfgScale).toBe(7);
    expect(result.changes.map(({ subject }) => subject)).not.toContain('guidance');
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

describe('SDXL Refiner policy', () => {
  const refiner: MainModelConfig = { base: 'sdxl-refiner', key: 'refiner', name: 'SDXL Refiner', type: 'main' };

  it('offers the refiner section for SDXL only and never lists refiners as main models', () => {
    expect(getGenerationUiPolicy(createModel('sdxl')).refinerVisible).toBe(true);
    expect(getGenerationUiPolicy(createModel('sd-1')).refinerVisible).toBe(false);
    expect(getGenerationUiPolicy(createModel('flux')).refinerVisible).toBe(false);
    expect(isGenerateModelSelectable(refiner)).toBe(false);
    expect(isSupportedGenerateModel(refiner)).toBe(false);
  });

  it('keeps the refiner and its tuning when leaving SDXL, idle until the next SDXL pick', () => {
    const flux = createModel('flux');
    const result = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sdxl'), { refinerModel: refiner, refinerStart: 0.6 }),
      model: flux,
      models: [],
    });

    expect(result.settings.refinerModel).toEqual(refiner);
    expect(result.settings.refinerStart).toBe(0.6);
    expect(result.changes).toEqual([]);
    expect(getActiveRefiner(result.settings, flux)).toBeNull();

    const kept = getGenerateModelSelectionResult({
      currentValues: createSettings(createModel('sdxl'), { refinerModel: refiner }),
      model: createModel('sdxl', { key: 'sdxl-other' }),
      models: [],
    });

    expect(kept.settings.refinerModel).toEqual(refiner);
    expect(kept.changes).toEqual([]);
  });

  it('validates the refiner only on SDXL, rejecting PiD and out-of-range tuning', () => {
    const sdxl = createModel('sdxl');
    const withRefiner = (overrides: Partial<GenerateSettings> = {}) =>
      getGenerationValidationReasons(sdxl, createSettings(sdxl, { refinerModel: refiner, ...overrides }));

    expect(withRefiner()).toEqual([]);
    expect(withRefiner({ refinerStart: 1.2, refinerSteps: 0 })).toEqual([
      'Refiner steps must be a whole number of at least 1.',
      'Refiner start must be between 0 and 1.',
    ]);
    expect(withRefiner({ refinerNegativeAestheticScore: 0 })).toEqual([
      'Refiner negative aesthetic score must be between 1 and 10.',
    ]);
    expect(
      withRefiner({
        gemma2EncoderModel: { base: 'any', key: 'gemma2', name: 'Gemma 2', type: 'gemma2_encoder' },
        pidDecoderModel: { base: 'sdxl', key: 'pid', name: 'PiD SDXL', type: 'pid_decoder' },
        pidMode: 'fit',
      })
    ).toContain('PiD decode is not compatible with the SDXL Refiner. Turn PiD off or clear the refiner model.');
    expect(withRefiner({ refinerModel: sdxl })).toEqual([
      `${sdxl.name} is not an SDXL Refiner model. Pick a refiner or clear the slot.`,
    ]);

    const sd1 = createModel('sd-1');

    expect(
      getGenerationValidationReasons(sd1, createSettings(sd1, { refinerModel: refiner, refinerStart: 5 }))
    ).toEqual([]);
  });

  it('ignores unavailable models and invalid tuning while the refiner is off, then validates reactivation', () => {
    const sdxl = createModel('sdxl');
    const settings = createSettings(sdxl, {
      refinerModel: refiner,
      refinerStart: 1,
      refinerSteps: 0,
      refinerCfgScale: Number.NaN,
      refinerPositiveAestheticScore: 0,
      refinerNegativeAestheticScore: 20,
    });

    expect(getGenerationValidationReasons(sdxl, settings)).toEqual([]);
    expect(getGenerationModelAvailabilityReasons(sdxl, settings, [sdxl as never])).toEqual([]);
    const reactivated = { ...settings, refinerStart: 0.8 };
    expect(getGenerationValidationReasons(sdxl, reactivated)).toContain(
      'Refiner steps must be a whole number of at least 1.'
    );
    expect(getGenerationModelAvailabilityReasons(sdxl, reactivated, [sdxl as never])).toEqual([
      'Refiner "SDXL Refiner" is not installed. Choose another, or clear it.',
    ]);
  });

  it('reports an uninstalled refiner', () => {
    const sdxl = createModel('sdxl');
    const settings = createSettings(sdxl, { refinerModel: refiner });

    expect(getGenerationModelAvailabilityReasons(sdxl, settings, [sdxl as never])).toEqual([
      'Refiner "SDXL Refiner" is not installed. Choose another, or clear it.',
    ]);
    expect(getGenerationModelAvailabilityReasons(sdxl, settings, [sdxl as never, refiner as never])).toEqual([]);
  });
});

describe('generation issues', () => {
  const englishCatalog = Object.values(
    import.meta.glob('../../../../public/locales/en.json', { eager: true, import: 'default' })
  )[0] as Record<string, unknown>;
  const ipAdapterReference = (id: string, overrides: Record<string, unknown> = {}): GenerateReferenceImage => ({
    config: {
      beginEndStepPct: [0, 1],
      clipVisionModel: 'ViT-H',
      image: referenceImage,
      method: 'full',
      model: sdxlIpAdapter,
      type: 'ip_adapter',
      weight: 1,
      ...overrides,
    } as GenerateReferenceImage['config'],
    id,
    isEnabled: true,
  });
  const sdxl = createModel('sdxl');

  const withoutPluralSuffix = (key: string) => key.replace(/_(one|other)$/, '');
  const localeGroup = (name: string) =>
    (englishCatalog.widgets as Record<string, Record<string, Record<string, string>>>).generate![name]!;

  it('words every translated issue in its locale entry exactly as its English, and the locale has no others', async () => {
    const instance = i18n.createInstance();
    await instance.init({
      initAsync: false,
      interpolation: { escapeValue: false },
      lng: 'en',
      resources: { en: { translation: englishCatalog } },
    });
    const translated = collectGenerationIssueScenarios().filter((issue) => issue.messageKey);

    expect(new Set(translated.map(({ messageKey }) => messageKey!.replace('widgets.generate.issues.', '')))).toEqual(
      new Set(Object.keys(localeGroup('issues')).map(withoutPluralSuffix))
    );

    for (const issue of translated) {
      expect(describeGenerationIssue((key, values) => instance.t(key, values), issue)).toBe(issue.message);
    }
  });

  it('names every subject a switch can report, and no others, in the locale', () => {
    expect(new Set(Object.keys(localeGroup('switchSubjects')).map(withoutPluralSuffix))).toEqual(
      new Set(GENERATE_MODEL_SWITCH_SUBJECTS)
    );
  });

  it('names each issue the section that fixes it and keeps its id while the problem lasts', () => {
    const settings = createSettings(sdxl, {
      referenceImages: Array.from({ length: 7 }, (_, index) => ipAdapterReference(`ref-${index}`)),
    });
    const [limit] = getGenerationValidationIssues(sdxl, settings);

    expect(limit).toMatchObject({
      id: 'referenceImages:limit',
      target: { field: 'referenceImages', section: 'guidance' },
    });
    expect(getGenerationValidationIssues(sdxl, { ...settings, steps: 12 })[0]?.id).toBe(limit?.id);
  });

  it('fixes the reference limit by turning off the extra running references, keeping them listed', () => {
    const referenceImages = [
      { ...ipAdapterReference('off'), isEnabled: false },
      ...Array.from({ length: 7 }, (_, index) => ipAdapterReference(`ref-${index}`)),
    ];
    const settings = createSettings(sdxl, { referenceImages });
    const [limit] = getGenerationValidationIssues(sdxl, settings);
    const fixed = { ...settings, ...getGenerationIssueFixPatch(settings, limit!.fix!) };

    expect(fixed.referenceImages.map(({ id, isEnabled }) => [id, isEnabled])).toEqual([
      ['off', false],
      ['ref-0', true],
      ['ref-1', true],
      ['ref-2', true],
      ['ref-3', true],
      ['ref-4', true],
      ['ref-5', false],
      ['ref-6', false],
    ]);
    expect(getGenerationValidationIssues(sdxl, fixed)).toEqual([]);
  });

  it('fixes an off-grid free size by snapping both sides, and recording the new ratio', () => {
    const settings = createSettings(sdxl, { aspectRatioId: 'Free', height: 1001, width: 1024 });
    const [grid] = getGenerationValidationIssues(sdxl, settings);

    expect(grid).toMatchObject({
      fix: { aspectRatioValue: 1024 / 1000, height: 1000, kind: 'setDimensions', width: 1024 },
      target: { section: 'dimensions' },
    });
    expect(
      getGenerationValidationIssues(sdxl, { ...settings, ...getGenerationIssueFixPatch(settings, grid!.fix!) })
    ).toEqual([]);
  });

  it('fixes an off-grid size under a ratio lock by deriving the height from the width', () => {
    const settings = createSettings(sdxl, {
      aspectRatioId: '16:9',
      aspectRatioValue: 16 / 9,
      height: 577,
      width: 1024,
    });
    const [grid] = getGenerationValidationIssues(sdxl, settings);

    expect(grid?.fix).toEqual({ height: 576, kind: 'setDimensions', width: 1024 });
  });

  it('fits a provider maximum on the grid, keeping a locked ratio', () => {
    const model: GenerateModelConfig = {
      ...externalModel,
      capabilities: { max_image_size: { height: 1000, width: 1000 }, modes: ['txt2img'] },
    };
    const settings = createSettings(model, { aspectRatioId: '1:1', aspectRatioValue: 1, height: 1500, width: 1500 });
    const fix = getGenerationValidationIssues(model, settings).find(({ id }) => id === 'dimensions:providerMax')?.fix;
    const grid = getGenerationDimensions(model).grid;

    expect(fix).toMatchObject({ kind: 'setDimensions' });
    expect(fix?.kind === 'setDimensions' && fix.width).toBeLessThanOrEqual(1000);
    expect(fix?.kind === 'setDimensions' && fix.width % grid).toBe(0);
    expect(fix?.kind === 'setDimensions' && fix.height).toBe(fix?.kind === 'setDimensions' && fix.width);
  });

  it('checks a Canvas hosted provider at the size it is asked for, as the form and the route both do', () => {
    const model: GenerateModelConfig = {
      ...externalModel,
      capabilities: {
        aspect_ratio_sizes: { '1:1': { height: 1024, width: 1024 } },
        max_image_size: { height: 1024, width: 1024 },
        modes: ['txt2img'],
      },
    };
    const values = { ...createSettings(model, { height: 3000, width: 3000 }), model };

    expect(getGenerateInvocationIssues(values).map(({ id }) => id)).toContain('dimensions:providerMax');
    expect(getGenerateInvocationIssues(values, { canvasFrame: { height: 3000, width: 3000 } })).toEqual([]);
  });
});
