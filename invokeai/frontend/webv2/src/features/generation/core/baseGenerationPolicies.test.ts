import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities.testing';
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
  addReferenceImages,
  getAutoFlux2ComponentSourceModel,
  pickFallbackGenerateModel,
  coerceSchedulerForGraph,
  getComponentPolicyContext,
  getComponentSectionPolicy,
  getDefaultComponentSlotPolicies,
  getDefaultGenerateSettings,
  getGenerateInvocationIssues,
  getGenerateModelSelectionResult,
  getGenerationDefaults,
  getGenerationDimensions,
  getGuidanceBoundReason,
  getGuidanceSliderMax,
  getGenerationModelAvailabilityReasons,
  getGenerationUiPolicy,
  getGenerationValidationIssues,
  getGenerationValidationReasons,
  getCompatibleReferenceImages,
  getModelDefaultComponents,
  getModelDefaultFills,
  pruneClosedDefaultComponents,
  getReferenceImageRoom,
  getReferenceImageUnavailableReason,
  getPromptPolicy,
  getSchedulerOptions,
  resolveModelDefaultVae,
  isReferenceImageSupported,
  isComponentCompatibleWithModel,
  isGenerateModelSelectable,
  isLossyModelSwitch,
  isModelFromRepo,
  isSupportedGenerateModel,
  needsEmptyCanvasFrameForReferenceBudget,
  type NewReferenceImage,
  type ReferenceVisionComponents,
} from './baseGenerationPolicies';
import { getGenerationIssueFixPatch } from './generationIssues';
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

const addTo = (
  model: GenerateModelConfig,
  models: readonly ComponentModelConfig[],
  referenceImages: GenerateReferenceImage[],
  additions: readonly NewReferenceImage[]
) => addReferenceImages({ additions, components: {}, mode: 'strict', model, models, referenceImages }).referenceImages;

seedArchitectureCapabilities();

describe('architecture policy, read from the backend capability table', () => {
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

  describe('the FLUX.3 prompt upsampler component', () => {
    const klein = createModel('flux3', { format: 'diffusers', variant: 'flux3_klein' });
    const llm = (overrides: Record<string, unknown> = {}) => ({
      base: 'any',
      key: 'pup',
      name: 'FLUX.3 Prompt Upsampler',
      supports_images: true,
      type: 'text_llm',
      ...overrides,
    });
    const slotFor = (settings: GenerateSettings) =>
      getComponentSectionPolicy(klein, settings).slots.find((slot) => slot.key === 'flux3PromptUpsamplerModel');
    const issues = (settings: GenerateSettings) =>
      getGenerationValidationIssues(klein, settings).filter(
        (issue) => issue.target.field === 'flux3PromptUpsamplerModel'
      );
    it('always has a slot, required once the switch is on, and then only for a model that reads images', () => {
      const off = createSettings(klein);

      expect(slotFor(off)?.required?.(getComponentPolicyContext(klein, off))).toBe(false);
      expect(issues(off)).toEqual([]);

      const on = createSettings(klein, { flux3PromptUpsamplerEnabled: true });

      expect(slotFor(on)?.required?.(getComponentPolicyContext(klein, on))).toBe(true);
      expect(issues(on)).toEqual([
        expect.objectContaining({
          message: 'Generate needs a Prompt Upsampler model while Prompt upsampler is on.',
          target: { field: 'flux3PromptUpsamplerModel', section: 'components' },
        }),
      ]);
      expect(issues({ ...on, flux3PromptUpsamplerModel: llm() })).toEqual([]);
      expect(issues({ ...on, flux3PromptUpsamplerModel: llm({ supports_images: false }) })).toHaveLength(1);
      expect(issues({ ...on, flux3PromptUpsamplerModel: llm({ type: 'llava_onevision' }) })).toHaveLength(1);
      expect(issues({ ...on, flux3PromptUpsamplerEnabled: false })).toEqual([]);
    });
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

  describe('isGenerateModelSelectable', () => {
    const unconditional = createModel('ideogram-4', { branch: 'unconditional', format: 'checkpoint' });
    const conditional = createModel('ideogram-4', { branch: 'conditional', format: 'checkpoint' });

    it('keeps a model that cannot run on its own out of every picker', () => {
      expect(isSupportedGenerateModel(unconditional)).toBe(true);
      expect(isGenerateModelSelectable(unconditional)).toBe(false);

      expect(isGenerateModelSelectable(conditional)).toBe(true);
      expect(isGenerateModelSelectable(createModel('ideogram-4', { format: 'diffusers' }))).toBe(true);
      expect(isGenerateModelSelectable(createModel('wan', { variant: 'a14b' }))).toBe(true);
    });

    it('starts on a selectable main model whose files are present, never on a standalone branch', () => {
      const gone = { ...createModel('sdxl', { key: 'gone' }), is_missing: true } as MainModelConfig;
      const sdxl = createModel('sdxl', { key: 'sdxl-ok' });

      expect(pickFallbackGenerateModel([gone, unconditional, sdxl])).toBe(sdxl);
      // All unusable by file: the first selectable one, not the branch.
      expect(pickFallbackGenerateModel([unconditional, gone])).toBe(gone);
      expect(pickFallbackGenerateModel([unconditional])).toBeUndefined();
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

  it('validates FLUX.2 Qwen3 and VAE requirements and allows bundled alternatives', () => {
    const model = createModel('flux2', { format: 'gguf_quantized', variant: 'klein_9b' });
    const source = createModel('flux2', { format: 'diffusers', variant: 'klein_9b' });
    const incompatibleSource = createModel('flux2', {
      format: 'diffusers',
      key: 'flux2-4b-source',
      variant: 'klein_4b',
    });

    expect(getGenerationValidationReasons(model, createSettings(model))).toEqual([
      'Generate needs a Qwen3 8B Encoder or a same-size FLUX.2 Klein Diffusers or SDNQ pipeline for this FLUX.2 Klein model.',
      'Generate needs a FLUX.2 VAE or a FLUX.2 Diffusers or SDNQ pipeline for this FLUX.2 Klein model.',
    ]);
    expect(getGenerationValidationReasons(model, createSettings(model, { componentSourceModel: source }))).toEqual([]);
    expect(
      getGenerationValidationReasons(model, createSettings(model, { componentSourceModel: incompatibleSource }))
    ).toEqual([
      'Generate needs a Qwen3 8B Encoder or a same-size FLUX.2 Klein Diffusers or SDNQ pipeline for this FLUX.2 Klein model.',
    ]);
    expect(
      getGenerationValidationReasons(model, createSettings(model, { qwen3EncoderModel: qwen3Encoder, vae: flux2Vae }))
    ).toEqual([]);
  });

  it('picks no FLUX.2 component source whose files are missing, whatever the catalog order', () => {
    const klein = createModel('flux2', { format: 'checkpoint', key: 'klein-single', variant: 'klein_9b' });
    const pipeline = (key: string, missing: boolean) => ({
      ...createModel('flux2', { format: 'diffusers', key, name: key, variant: 'klein_9b' }),
      ...(missing ? { is_missing: true } : {}),
    });
    const gone = pipeline('pipeline-gone', true);
    const working = pipeline('pipeline-ok', false);
    const pick = (models: unknown[]) =>
      getAutoFlux2ComponentSourceModel(
        klein,
        { mistralEncoderModel: null, qwen3EncoderModel: null, vae: null },
        models as never
      );

    expect(pick([gone, working])?.key).toBe('pipeline-ok');
    expect(pick([working, gone])?.key).toBe('pipeline-ok');
    expect(pick([gone])).toBeNull();
  });

  describe('FLUX.2 [dev] component sources', () => {
    const model = createModel('flux2', { format: 'gguf_quantized', variant: 'dev' });
    const completePipeline = { text_encoder: {}, tokenizer: {}, transformer: {}, vae: {} };
    const sdnqKlein = createModel('flux2', {
      format: 'sdnq_quantized',
      key: 'flux2-klein-sdnq',
      submodels: completePipeline,
      variant: 'klein_9b',
    });
    const vaeMissing = 'Generate needs a FLUX.2 VAE or a FLUX.2 Diffusers pipeline for this FLUX.2 [dev] model.';
    const selectDev = (currentValues: GenerateSettings, models: MainModelConfig[]) =>
      getGenerateModelSelectionResult({ currentValues, model, models }).settings;

    it('drops a saved or recalled SDNQ pipeline source', () => {
      const settings = createSettings(model, { componentSourceModel: sdnqKlein, mistralEncoderModel: mistralEncoder });

      expect(isComponentCompatibleWithModel(model, settings, 'componentSourceModel', sdnqKlein)).toBe(false);
      expect(getGenerationValidationReasons(model, settings)).toEqual([vaeMissing]);
      expect(selectDev(settings, [sdnqKlein]).componentSourceModel).toBeNull();
    });
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

  describe('Qwen Image Edit with a GGUF Qwen VL Encoder', () => {
    const model = createModel('qwen-image', { format: 'gguf_quantized', variant: 'edit' });
    const ggufEncoder: ComponentModelConfig = {
      ...qwenVlEncoder,
      format: 'gguf_quantized',
      key: 'qwen-vl-gguf',
      name: 'Qwen VL Q4_K_M',
    };
    const reference = (id: string, isEnabled = true) => ({
      config: { image: referenceImage, type: 'qwen_image_reference_image' as const },
      id,
      isEnabled,
    });
    const reasonsFor = (overrides: Partial<GenerateSettings>) =>
      getGenerationValidationReasons(
        model,
        createSettings(model, { qwenVLEncoderModel: ggufEncoder, vae: qwenImageVae, ...overrides })
      );

    it('blocks Invoke once with the reason when reference images are on', () => {
      expect(reasonsFor({ referenceImages: [reference('a'), reference('b')] })).toEqual([
        'Qwen VL Encoder "Qwen VL Q4_K_M" is a GGUF build with no vision tower, so it cannot read reference images. ' +
          'Select a safetensors or Diffusers Qwen VL Encoder, or turn the reference images off.',
      ]);
    });
  });

  describe('Z-Image Qwen3 encoder', () => {
    const model = createModel('z-image', { format: 'checkpoint' });
    const qwen3_4b: ComponentModelConfig = { ...qwen3Encoder, key: 'qwen3-4b', name: 'Qwen3 4B', variant: 'qwen3_4b' };
    const qwen3_8b: ComponentModelConfig = { ...qwen3Encoder, key: 'qwen3-8b', name: 'Qwen3 8B', variant: 'qwen3_8b' };
    const source = createModel('z-image', { key: 'z-image-diffusers', format: 'diffusers' });
    const incompatible =
      'The selected Qwen3 Encoder is not the 4B one Z-Image conditions on. Select a Qwen3 4B Encoder or clear it.';

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

  it('requires all four SD3 components for a GGUF transformer and none for a Diffusers pipeline', () => {
    const gguf = createModel('sd-3', { format: 'gguf_quantized' });
    const diffusers = createModel('sd-3', { format: 'diffusers' });
    const clipG = { ...clipEmbed, key: 'clip-g', variant: 'gigantic' };
    const sd3Vae: VaeModelConfig = { base: 'sd-3', key: 'sd3-vae', name: 'SD3 VAE', type: 'vae' };

    expect(getGenerationValidationReasons(gguf, createSettings(gguf))).toEqual([
      'Generate needs a T5 Encoder for non-Diffusers SD3 models.',
      'Generate needs a CLIP L encoder for non-Diffusers SD3 models.',
      'Generate needs a CLIP G encoder for non-Diffusers SD3 models.',
      'Generate needs a VAE for non-Diffusers SD3 models.',
    ]);
    expect(
      getGenerationValidationReasons(
        gguf,
        createSettings(gguf, {
          clipGEmbedModel: clipG,
          clipLEmbedModel: clipG,
          t5EncoderModel: t5Encoder,
          vae: fluxVae,
        })
      )
    ).toEqual([
      'Generate needs a CLIP L encoder for non-Diffusers SD3 models.',
      'Generate needs a VAE for non-Diffusers SD3 models.',
    ]);
    expect(
      getGenerationValidationReasons(
        gguf,
        createSettings(gguf, {
          clipGEmbedModel: clipG,
          clipLEmbedModel: clipEmbed,
          t5EncoderModel: t5Encoder,
          vae: sd3Vae,
        })
      )
    ).toEqual([]);
    expect(getGenerationValidationReasons(diffusers, createSettings(diffusers))).toEqual([]);
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

  it('warns when only visible Canvas content would exceed the reference budget', () => {
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
    const values = { ...settings, model };
    expect(needsEmptyCanvasFrameForReferenceBudget(model, settings)).toBe(true);
    expect(getGenerateInvocationIssues(values, { canvasFrame: { height: 512, width: 512 } })).toEqual([]);
    expect(getGenerateInvocationIssues(values)).toEqual([]);
    expect(getGenerationValidationReasons(model, settings, 'img2img')).toEqual([
      'External Model supports at most 15 images total (reference + init + output), got 16.',
    ]);
    expect(needsEmptyCanvasFrameForReferenceBudget({ ...model, capabilities: { modes: ['txt2img'] } }, settings)).toBe(
      false
    );
    settings.referenceImages[0]!.isEnabled = false;
    expect(needsEmptyCanvasFrameForReferenceBudget(model, settings)).toBe(false);
    settings.referenceImages.push(
      { ...settings.referenceImages[1]!, id: 'extra-1' },
      { ...settings.referenceImages[2]!, id: 'extra-2' }
    );
    expect(needsEmptyCanvasFrameForReferenceBudget(model, settings)).toBe(false);
    expect(getGenerateInvocationIssues({ ...settings, model }).map(({ message }) => message)).toContain(
      'External Model supports at most 15 images total (reference + init + output), got 16.'
    );
  });

  it('blocks FLUX Fill outside inpaint and outpaint, but not while a Canvas frame decides the mode', () => {
    const model = createModel('flux', { variant: 'dev_fill' });
    const settings = {
      ...createSettings(model),
      clipEmbedModel: clipEmbed,
      t5EncoderModel: t5Encoder,
      vae: fluxVae,
    };
    const reasons = (mode?: Parameters<typeof getGenerationValidationReasons>[2]) =>
      getGenerationValidationReasons(model, settings, mode).filter((reason) => reason.startsWith('FLUX Fill only'));

    expect(reasons()).toHaveLength(1);
    expect(reasons('img2img')).toHaveLength(1);
    expect(reasons('inpaint')).toEqual([]);
    expect(reasons('outpaint')).toEqual([]);
    expect(getGenerateInvocationIssues({ ...settings, model }).map(({ message }) => message)).toEqual(reasons());
    expect(
      getGenerateInvocationIssues({ ...settings, model }, { canvasFrame: { height: 512, width: 512 } }).map(
        ({ message }) => message
      )
    ).not.toContain(reasons()[0]);
  });

  it('holds FLUX Fill to a guidance floor of 25 with the field reason and a one-step fix', () => {
    const model = createModel('flux', { variant: 'dev_fill' });
    const settings = { ...createSettings(model), clipEmbedModel: clipEmbed, t5EncoderModel: t5Encoder, vae: fluxVae };
    const guidanceIssues = (cfgScale: number) =>
      getGenerationValidationIssues(model, { ...settings, cfgScale }, 'inpaint').filter(
        ({ target }) => target.field === 'cfgScale'
      );

    expect(getGuidanceBoundReason(model, 3.5)).toBe(`Guidance must be at least 25 for ${model.name}.`);
    expect(guidanceIssues(3.5)).toEqual([
      expect.objectContaining({
        fix: { kind: 'clampGuidance', value: 25 },
        message: getGuidanceBoundReason(model, 3.5),
      }),
    ]);
    expect(getGuidanceBoundReason(model, 25)).toBeNull();
    expect(guidanceIssues(25)).toEqual([]);
    expect(guidanceIssues(30)).toEqual([]);
    expect(getGenerationUiPolicy(model).guidanceMin).toBe(25);
  });

  it('lands on the FLUX Fill floor when switching from FLUX dev at its default guidance', () => {
    const dev = createModel('flux', { key: 'flux-dev', variant: 'dev' });
    const fill = createModel('flux', { key: 'flux-fill', variant: 'dev_fill' });
    const currentValues = createSettings(dev, { cfgScale: 3.5 });

    const { settings } = getGenerateModelSelectionResult({ currentValues, model: fill, models: [dev, fill] });

    expect(settings.cfgScale).toBeGreaterThanOrEqual(25);
  });

  it('raises a FLUX Fill model default saved below the floor to the floor', () => {
    const fill = createModel('flux', { variant: 'dev_fill', default_settings: { guidance: 3.5 } });

    expect(getGenerationDefaults(fill).cfgScale).toBe(25);
    expect(
      getGenerationDefaults(createModel('flux', { variant: 'dev', default_settings: { guidance: 3.5 } })).cfgScale
    ).toBe(3.5);
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

    expect(toFlux.settings.loras).toBe(loras);
    expect(toFlux.changes).toEqual([{ count: 1, kind: 'turnedOff', subject: 'concepts' }]);
    expect(isLossyModelSwitch(toFlux)).toBe(false);

    const back = getGenerateModelSelectionResult({ currentValues: toFlux.settings, model: sdxl, models: [sdxl, flux] });

    expect(back.settings.loras).toBe(loras);
    expect(back.changes).toEqual([]);
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

  it('reconciles dimensions to the new model grid as an adjustment, not a removed setting', () => {
    const model = createModel('cogview4');
    const settings = createSettings(createModel('sdxl'), { height: 520, width: 520 });
    const result = getGenerateModelSelectionResult({ currentValues: settings, model, models: [] });

    expect(result.settings.width).toBe(512);
    expect(result.settings.height).toBe(512);
    expect(result.changes).toEqual([]);
    expect(result.dimensionsAdjusted).toBe(true);
  });

  describe('a default VAE', () => {
    const fluxVae: VaeModelConfig = { base: 'flux', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' };
    const resolve = (
      base: string,
      vae: string | null | undefined,
      vaeModels: VaeModelConfig[] = [fluxVae],
      format?: MainModelConfig['format']
    ) => {
      const model = createModel(base, { format, ...(vae === undefined ? {} : { default_settings: { vae } }) });

      return resolveModelDefaultVae(model, createSettings(model), vaeModels);
    };

    it('applies one that is installed and fits, by shared VAE compatibility', () => {
      expect(resolve('z-image', 'flux-vae')).toEqual({ kind: 'apply', vae: fluxVae });
      expect(resolve('sdxl', undefined)).toEqual({ kind: 'none' });
      expect(resolve('sdxl', null)).toEqual({ kind: 'none' });
    });
  });

  it('reports selected model records that disappeared from the backend model list', () => {
    const model = createModel('flux');
    const settings = createSettings(model, {
      clipEmbedModel: clipEmbed,
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
    it('types a new Krea-2 reference as style only when it is the one running reference', () => {
      const typesAfterAdding = (count: number, referenceImages: GenerateSettings['referenceImages'] = []) =>
        addTo(
          krea2,
          [],
          referenceImages,
          Array.from({ length: count }, (_, index) => ({ id: `new-${index}`, image: referenceImage }))
        ).map(({ config, id, isEnabled }) => [id, config.type, isEnabled]);

      expect(typesAfterAdding(1)).toEqual([['new-0', 'krea2_reference_image', true]]);
      expect(typesAfterAdding(3).map(([, type]) => type)).toEqual([
        'krea2_edit_reference_image',
        'krea2_edit_reference_image',
        'krea2_edit_reference_image',
      ]);
      expect(typesAfterAdding(1, [edit('a')])).toEqual([
        ['a', 'krea2_edit_reference_image', true],
        ['new-0', 'krea2_edit_reference_image', true],
      ]);
      const bypassed = { ...style('s'), config: { ...style('s').config, styleStrength: 0 } };

      expect(typesAfterAdding(1, [edit('a', false), bypassed])).toEqual([
        ['a', 'krea2_edit_reference_image', false],
        ['s', 'krea2_reference_image', true],
        ['new-0', 'krea2_reference_image', true],
      ]);
    });

    it('runs one enabled style reference at a time and never mixes style with edit references', () => {
      const styleReason =
        'Krea-2 uses one style reference at a time. Turn off all but one, or switch all of them to Edit.';
      const mixReason =
        'Krea-2 cannot combine a style reference with edit references. Switch the style reference to Edit, or turn off one kind.';

      expect(reasonsFor(krea2, { referenceImages: [style('a'), style('b')] })).toEqual([styleReason]);
      expect(reasonsFor(krea2, { referenceImages: [style('a'), edit('b')] })).toEqual([mixReason]);
      expect(reasonsFor(krea2, { referenceImages: [style('a', false), style('b'), edit('c', false)] })).toEqual([]);
      expect(reasonsFor(krea2, { referenceImages: [style('a', false), edit('b'), edit('c')] })).toEqual([]);
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

  describe('Qwen-Image-2.1 references follow the encoder that reads them', () => {
    const encoderModel = (format: string) =>
      ({
        base: 'any',
        format,
        key: `qwen3-vl-8b-${format}`,
        name: 'Qwen3-VL 8B',
        type: 'qwen3_vl_encoder',
        variant: 'qwen3_vl_8b',
      }) as const;
    const encoder = (format: string) => ({ qwen3VLEncoderModel: encoderModel(format) });
    it('reads them with the standalone safetensors encoder of a single-file main', () => {
      const checkpoint = createModel('qwen-image-2-1', { format: 'checkpoint' });
      const gguf = createModel('qwen-image-2-1', { format: 'gguf_quantized' });

      expect(isReferenceImageSupported(checkpoint, encoder('checkpoint'))).toBe(true);
      expect(isReferenceImageSupported(gguf, encoder('checkpoint'))).toBe(true);
      expect(isReferenceImageSupported(checkpoint, encoder('qwen3_vl_encoder'))).toBe(true);
      expect(getReferenceImageRoom(checkpoint, encoder('checkpoint'), []).limit).toBe(4);
      expect(getReferenceImageUnavailableReason(checkpoint, encoder('checkpoint'))).toBeNull();
    });

    it('never lets an encoder change delete references on a model switch', () => {
      const checkpoint = createModel('qwen-image-2-1', { format: 'checkpoint' });
      const referenceImages = [
        {
          config: { image: referenceImage, type: 'qwen_image_2_1_reference_image' as const },
          id: 'ref-1',
          isEnabled: true,
        },
      ];

      expect(getCompatibleReferenceImages(referenceImages, checkpoint, [])).toBe(referenceImages);
    });
  });

  it('derives the reference image limit from external provider capabilities', () => {
    expect(getReferenceImageRoom(createModel('sdxl'), {}, []).limit).toBe(5);
    expect(getReferenceImageRoom(createModel('cogview4'), {}, []).limit).toBe(0);
    expect(
      getReferenceImageRoom(
        {
          ...externalModel,
          capabilities: { ...externalModel.capabilities, max_reference_images: 14, supports_reference_images: true },
        },
        {},
        []
      ).limit
    ).toBe(14);
    expect(
      getReferenceImageRoom(
        { ...externalModel, capabilities: { ...externalModel.capabilities, supports_reference_images: true } },
        {},
        []
      ).limit
    ).toBe(5);
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
    ).toContain('sd-3 model cannot use Reference image 1. Turn it off, or remove it.');
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
});

describe('adding reference images', () => {
  const sdxl = createModel('sdxl');
  const krea2 = createModel('krea-2', { format: 'diffusers' });
  const qwen21 = createModel('qwen-image-2-1', { format: 'checkpoint' });
  const images = (...ids: string[]): NewReferenceImage[] => ids.map((id) => ({ id, image: referenceImage }));
  const adapter = (id: string, model: ComponentModelConfig = sdxlIpAdapter): GenerateReferenceImage => ({
    config: {
      beginEndStepPct: [0, 1],
      clipVisionModel: 'ViT-H',
      image: referenceImage,
      method: 'full',
      model,
      type: 'ip_adapter',
      weight: 1,
    },
    id,
    isEnabled: true,
  });
  const style = (id: string): GenerateReferenceImage => ({
    config: { image: referenceImage, styleStrength: 1, type: 'krea2_reference_image' },
    id,
    isEnabled: true,
  });
  const edit = (id: string, isEnabled = true): GenerateReferenceImage => ({
    config: { image: referenceImage, type: 'krea2_edit_reference_image' },
    id,
    isEnabled,
  });
  const strict = (
    model: GenerateModelConfig,
    referenceImages: GenerateReferenceImage[],
    additions: NewReferenceImage[],
    components: ReferenceVisionComponents = {}
  ) => addReferenceImages({ additions, components, mode: 'strict', model, models: [sdxlIpAdapter], referenceImages });
  const keep = (
    model: GenerateModelConfig,
    referenceImages: GenerateReferenceImage[],
    additions: NewReferenceImage[],
    recalled: GenerateReferenceImage[] = []
  ) => addReferenceImages({ additions, mode: 'keep', model, models: [sdxlIpAdapter], recalled, referenceImages });

  it('places a user add in the free slots, or refuses it with a reason and leaves the list as it was', () => {
    const four = ['a', 'b', 'c', 'd'].map((id) => adapter(id));
    const partly = strict(sdxl, four, images('x', 'y', 'z'));

    expect(partly).toMatchObject({ added: 1, dropped: 2, limit: 5, refusal: null });
    expect(partly.referenceImages.map(({ id }) => id)).toEqual(['a', 'b', 'c', 'd', 'x']);

    const full = [...four, adapter('e')];

    expect(strict(sdxl, full, images('x'))).toEqual({
      added: 0,
      dropped: 1,
      limit: 5,
      referenceImages: full,
      refusal: 'full',
      unavailableReason: null,
    });
    expect(strict(createModel('cogview4'), [], images('x')).refusal).toBe('unsupported');
    expect(strict(qwen21, [], images('x'))).toMatchObject({ refusal: 'visionTower', unavailableReason: 'visionTower' });
    expect(
      strict(qwen21, [], images('x'), {
        qwen3VLEncoderModel: {
          base: 'any',
          format: 'gguf_quantized',
          key: 'qwen3-vl-8b-gguf',
          name: 'Qwen3-VL 8B',
          type: 'qwen3_vl_encoder',
          variant: 'qwen3_vl_8b',
        },
      })
    ).toMatchObject({
      added: 0,
      referenceImages: [],
      refusal: 'ggufVisionTower',
      unavailableReason: 'ggufVisionTower',
    });
  });

  it('keeps a recall whatever the encoder, retargets recalled adapters with the list, and cuts it to the limit', () => {
    // Validation flags the missing tower; recall keeps the references.
    expect(keep(qwen21, [], images('x'))).toMatchObject({
      added: 1,
      referenceImages: [{ config: { type: 'qwen_image_2_1_reference_image' }, id: 'x' }],
      refusal: null,
    });
    // Left as is, with nothing counted as dropped.
    const kept = [adapter('a')];

    expect(keep(createModel('cogview4'), kept, images('x'))).toEqual({
      added: 0,
      dropped: 0,
      limit: 0,
      referenceImages: kept,
      refusal: 'unsupported',
      unavailableReason: null,
    });

    const fluxAdapter = adapter('flux', { base: 'flux', key: 'flux-ip-adapter', name: 'FLUX IP', type: 'ip_adapter' });
    const recalled = keep(sdxl, [adapter('a'), adapter('b'), adapter('c')], images('x', 'y'), [fluxAdapter]);

    expect(recalled).toMatchObject({ added: 1, dropped: 1, limit: 5, refusal: null });
    expect(
      recalled.referenceImages.map(({ config, id }) => [id, config.type === 'ip_adapter' ? config.model?.key : null])
    ).toEqual([
      ['a', 'sdxl-ip-adapter'],
      ['b', 'sdxl-ip-adapter'],
      ['c', 'sdxl-ip-adapter'],
      ['flux', 'sdxl-ip-adapter'],
      ['x', 'sdxl-ip-adapter'],
    ]);
  });

  it('cuts additions past the limit before typing them, so a Krea-2 style reference stays when none lands', () => {
    const fullSet = [style('s'), edit('a', false), edit('b', false)];

    expect(keep(krea2, fullSet, images('x'))).toMatchObject({ dropped: 1, referenceImages: fullSet });
    expect(strict(krea2, fullSet, images('x'))).toMatchObject({ referenceImages: fullSet, refusal: 'full' });
    // A landing addition turns the lone style reference into edit.
    expect(strict(krea2, [style('s'), edit('a', false)], images('x', 'y')).referenceImages).toEqual([
      edit('s'),
      edit('a', false),
      edit('x'),
    ]);
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
});

describe('FLUX.3 policies', () => {
  const KLEIN_CFG_REASON = 'FLUX.3 Klein samples without CFG. Set CFG Scale to 1, or use FLUX.3 Klein Base for CFG.';

  it('blocks CFG on the step-distilled Klein and allows it on Klein Base', () => {
    const klein = createModel('flux3', { format: 'diffusers', variant: 'flux3_klein' });
    const kleinBase = createModel('flux3', { format: 'diffusers', variant: 'flux3_klein_base' });

    expect(getGenerationValidationReasons(klein, createSettings(klein))).toEqual([]);
    expect(getGenerationValidationReasons(klein, createSettings(klein, { cfgScale: 2 }))).toContain(KLEIN_CFG_REASON);
    expect(getGenerationValidationReasons(kleinBase, createSettings(kleinBase, { cfgScale: 3.5 }))).toEqual([]);
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
});

describe('Text LLM records that also serve as an encoder', () => {
  const textLlm = (roleType: string, roleVariant: string): ComponentModelConfig => ({
    base: 'any',
    format: 'checkpoint',
    key: `llm-${roleVariant}`,
    name: `LLM ${roleVariant}`,
    roles: [{ type: roleType, variant: roleVariant }],
    type: 'text_llm',
  });
  const offers = (model: MainModelConfig, slotKey: string, candidate: ComponentModelConfig): boolean | undefined => {
    const settings = createSettings(model);
    const slot = getComponentSectionPolicy(model, settings).slots.find((s) => s.key === slotKey);

    return slot?.filter?.(candidate, { model, selectedComponents: settings, settings });
  };

  it('offers a role-serving Text LLM only to the slot whose encoder size it matches', () => {
    const moss = createModel('moss-image', { format: 'checkpoint' });
    const zImage = createModel('z-image', { format: 'checkpoint' });
    const klein4b = createModel('flux2', { format: 'gguf_quantized', variant: 'klein_4b' });
    const anima = createModel('anima', { format: 'checkpoint', variant: 'anima_qwen3' });

    expect(offers(moss, 'qwen35EncoderModel', textLlm('qwen3_5_encoder', 'qwen3_5_9b'))).toBe(true);
    expect(offers(moss, 'qwen35EncoderModel', textLlm('qwen3_5_encoder', 'qwen3_5_4b'))).toBe(false);
    expect(offers(zImage, 'qwen3EncoderModel', textLlm('qwen3_encoder', 'qwen3_4b'))).toBe(true);
    expect(offers(zImage, 'qwen3EncoderModel', textLlm('qwen3_encoder', 'qwen3_8b'))).toBe(false);
    expect(offers(klein4b, 'qwen3EncoderModel', textLlm('qwen3_encoder', 'qwen3_4b'))).toBe(true);
    expect(offers(klein4b, 'qwen3EncoderModel', textLlm('qwen3_encoder', 'qwen3_8b'))).toBe(false);
    expect(offers(anima, 'qwen3EncoderModel', textLlm('qwen3_encoder', 'qwen3_06b'))).toBe(true);
    expect(offers(zImage, 'qwen3EncoderModel', { ...textLlm('qwen3_encoder', 'qwen3_4b'), roles: [] })).toBe(false);
  });

  it('counts a selected role-serving Text LLM as the missing encoder', () => {
    const moss = createModel('moss-image', { format: 'checkpoint' });
    const zImage = createModel('z-image', { format: 'checkpoint' });

    expect(
      getGenerationValidationReasons(
        moss,
        createSettings(moss, { qwen35EncoderModel: textLlm('qwen3_5_encoder', 'qwen3_5_9b'), vae: flux2Vae })
      )
    ).toEqual([]);
    expect(
      getGenerationValidationReasons(
        zImage,
        createSettings(zImage, { qwen3EncoderModel: textLlm('qwen3_encoder', 'qwen3_4b'), vae: fluxVae })
      )
    ).toEqual([]);
    expect(
      getGenerationValidationReasons(
        zImage,
        createSettings(zImage, { qwen3EncoderModel: textLlm('qwen3_encoder', 'qwen3_8b'), vae: fluxVae })
      )
    ).toEqual([
      'The selected Qwen3 Encoder is not the 4B one Z-Image conditions on. Select a Qwen3 4B Encoder or clear it.',
    ]);
  });
});

describe('Anima IP-Adapter references', () => {
  const animaQwen3 = createModel('anima', { format: 'checkpoint', variant: 'anima_qwen3' });
  const animaIpAdapter: ComponentModelConfig = {
    base: 'anima',
    key: 'anima-ip-adapter',
    name: 'Anima IP-Adapter',
    type: 'ip_adapter',
  };
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

  it('retargets an adapter-less reference only when a present IP-Adapter could take it', () => {
    const empty = { ...animaReference('ref-1'), config: { ...animaReference('ref-1').config, model: null } };
    const missing = { ...animaIpAdapter, is_missing: true } as ComponentModelConfig;
    const referenceImages = [empty];

    expect(getCompatibleReferenceImages(referenceImages, animaQwen3, [missing])).toBe(referenceImages);
    expect(getCompatibleReferenceImages(referenceImages, animaQwen3, [animaIpAdapter])).not.toBe(referenceImages);
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
  it('uses a 16px grid for all three, matching their transformer patch sizes', () => {
    expect(getGenerationDimensions(createModel('krea-2'))).toMatchObject({ grid: 16, optimal: 1024 });
    expect(getGenerationDimensions(createModel('ideogram-4'))).toMatchObject({ grid: 16, optimal: 1024 });
    expect(getGenerationDimensions(createModel('wan'))).toMatchObject({ grid: 16, optimal: 1024 });
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
});

describe('the guidance slider value from a model record', () => {
  it('reads the right field for every base in the table', () => {
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

  it('enforces the served bound for every architecture in the table', () => {
    for (const row of architectureCapabilitiesFixture) {
      const model = createModel(row.base, { variant: row.variant ?? undefined } as Partial<MainModelConfig>);

      // Ideogram's separate sampler override is validated with its own denoise bounds above.
      if (!isSupportedGenerateModel(model) || model.base === 'ideogram-4') {
        continue;
      }

      const where = `${row.base}/${row.variant ?? '-'}`;
      const { guidance_max: max } = row.features;
      // FLUX Fill is the one deliberate exception: a product floor above the served node minimum.
      const min = where === 'flux/dev_fill' ? 25 : row.features.guidance_min;

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
  const withoutBase = (base: string) => architectureCapabilitiesFixture.filter((row) => row.base !== base);

  it('blocks a supported base the backend did not describe', () => {
    const model = createModel('cogview4');
    const settings = createSettings(model);

    setArchitectureCapabilities(withoutBase('cogview4'));

    expect(getGenerationValidationReasons(model, settings)).toEqual([
      'The backend does not describe the cogview4 architecture, so it cannot be generated with.',
    ]);
  });
});

describe('getGenerationDimensions and the variant it dispatches on', () => {
  it('answers from the variant row when one differs, which its parameter type now admits', () => {
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
});

describe('generation issues', () => {
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
});

const REPO = 'google/gemma-4-E4B-it';

describe('isModelFromRepo', () => {
  it('matches the repo in its Hugging Face forms', () => {
    expect(isModelFromRepo('google/gemma-4-E4B-it', REPO)).toBe(true);
    expect(isModelFromRepo('Google/Gemma-4-e4b-IT', REPO)).toBe(true);
    expect(isModelFromRepo('google/gemma-4-E4B-it::text_encoder', REPO)).toBe(true);
    expect(isModelFromRepo('google/gemma-4-E4B-it:fp16::text_encoder', REPO)).toBe(true);
    expect(isModelFromRepo('https://huggingface.co/google/gemma-4-E4B-it/tree/main', REPO)).toBe(true);
  });

  it('refuses other repos, other folders and a missing source', () => {
    expect(isModelFromRepo('google/gemma-4-E2B-it', REPO)).toBe(false);
    expect(isModelFromRepo('someone/gemma-4-E4B-it', REPO)).toBe(false);
    expect(isModelFromRepo('someone/gemma-4-E4B-it:fp16', REPO)).toBe(false);
    expect(isModelFromRepo('https://example.test/google/gemma-4-E4B-it', REPO)).toBe(false);
    expect(isModelFromRepo('E:/ModelStuffSSD/gemma-4-E4B-it-q4', REPO)).toBe(false);
    expect(isModelFromRepo('E:/ModelStuffSSD/gemma-4-E4B-it/model.safetensors', REPO)).toBe(false);
    expect(isModelFromRepo(null, REPO)).toBe(false);
  });
});

describe('a stored FLUX.2 component source', () => {
  const qwen3Encoder: ComponentModelConfig = {
    base: 'any',
    key: 'qwen3',
    name: 'Qwen3 Encoder',
    type: 'qwen3_encoder',
    variant: 'qwen3_8b',
  };
  const flux2Vae: VaeModelConfig = { base: 'flux2', key: 'flux2-vae', name: 'FLUX.2 VAE', type: 'vae' };
  const model = createModel('flux2', { format: 'gguf_quantized', variant: 'klein_9b' });
  const picked = createModel('flux2', { format: 'diffusers', key: 'flux2-picked-source', variant: 'klein_9b' });
  const first = createModel('flux2', { format: 'diffusers', key: 'flux2-first-source', variant: 'klein_9b' });
  const smaller = createModel('flux2', { format: 'diffusers', key: 'flux2-4b-source', variant: 'klein_4b' });
  const settings = { ...createSettings(model), componentSourceModel: picked };

  it('stays while it is installed and gives the encoder the model lacks', () => {
    expect(getAutoFlux2ComponentSourceModel(model, settings, [first, picked])?.key).toBe(picked.key);
  });

  it('stays beside a standalone encoder, which needs only the VAE from it', () => {
    expect(
      getAutoFlux2ComponentSourceModel(
        model,
        { ...settings, componentSourceModel: smaller, qwen3EncoderModel: qwen3Encoder },
        [first, smaller]
      )?.key
    ).toBe(smaller.key);
  });

  it('is dropped once a standalone encoder and VAE need no source', () => {
    expect(
      getAutoFlux2ComponentSourceModel(model, { ...settings, qwen3EncoderModel: qwen3Encoder, vae: flux2Vae }, [picked])
    ).toBeNull();
  });

  it('is replaced when it is no longer installed', () => {
    expect(getAutoFlux2ComponentSourceModel(model, settings, [first])?.key).toBe(first.key);
  });

  it('is replaced when the standalone encoder is the wrong size and the source cannot give the encoder', () => {
    const wrongSize: ComponentModelConfig = { ...qwen3Encoder, key: 'qwen3-4b', variant: 'qwen3_4b' };

    expect(
      getAutoFlux2ComponentSourceModel(
        model,
        { ...settings, componentSourceModel: smaller, qwen3EncoderModel: wrongSize },
        [smaller, first]
      )?.key
    ).toBe(first.key);
  });

  it('is replaced when it cannot give the encoder the model needs', () => {
    expect(
      getAutoFlux2ComponentSourceModel(model, { ...settings, componentSourceModel: smaller }, [smaller, first])?.key
    ).toBe(first.key);
  });
});

describe("a main model's default components", () => {
  const withDefaults = (base: string, components: Record<string, string>, overrides: Partial<MainModelConfig> = {}) =>
    createModel(base, {
      default_settings: { components },
      format: 'checkpoint',
      key: `${base}-with-defaults`,
      ...overrides,
    });
  const switchTo = (model: GenerateModelConfig, models: unknown[], currentValues?: GenerateSettings) =>
    getGenerateModelSelectionResult({
      currentValues: currentValues ?? createSettings(createModel('sd-1')),
      model,
      models: models as ComponentModelConfig[],
    });
  const otherT5: ComponentModelConfig = { base: 'any', key: 't5-other', name: 'Other T5', type: 't5_encoder' };

  it('fills the empty slots on a switch without counting them as a loss', () => {
    const model = withDefaults('flux', { clip_embed_model: clipEmbed.key, t5_encoder: t5Encoder.key });
    const result = switchTo(model, [t5Encoder, clipEmbed]);

    expect(result.settings.t5EncoderModel).toBe(t5Encoder);
    expect(result.settings.clipEmbedModel).toBe(clipEmbed);
    expect(result.changes).toEqual([]);
    expect(isLossyModelSwitch(result)).toBe(false);
  });

  it('offers a Text LLM that serves as the encoder through a role in the Model Manager', () => {
    const model = createModel('z-image', { format: 'checkpoint' });
    const llm = {
      base: 'any',
      format: 'checkpoint',
      key: 'qwen3-llm',
      name: 'Qwen3 4B',
      roles: [{ type: 'qwen3_encoder', variant: 'qwen3_4b' }],
      type: 'text_llm',
    } as unknown as ComponentModelConfig;
    const slot = getDefaultComponentSlotPolicies(model, [llm]).find(
      (candidate) => candidate.persistenceKey === 'qwen3_encoder'
    );

    expect(slot?.filter(llm)).toBe(true);
  });

  it('takes no default whose files are missing on a switch, and replaces a pick that went missing', () => {
    const model = withDefaults('flux', { t5_encoder: t5Encoder.key });
    const missingT5 = { ...t5Encoder, is_missing: true };
    const picked = createSettings(createModel('sd-1'), { t5EncoderModel: otherT5 });

    expect(switchTo(model, [missingT5]).settings.t5EncoderModel).toBeNull();
    expect(switchTo(model, [missingT5, otherT5], picked).settings.t5EncoderModel).toBe(otherT5);
    expect(switchTo(model, [t5Encoder, { ...otherT5, is_missing: true }], picked).settings.t5EncoderModel).toBe(
      t5Encoder
    );
  });

  it('keeps a selection that already fits the new model', () => {
    const model = withDefaults('flux', { t5_encoder: t5Encoder.key });
    const current = createSettings(createModel('flux', { key: 'flux-other' }), { t5EncoderModel: otherT5 });

    expect(switchTo(model, [t5Encoder, otherT5], current).settings.t5EncoderModel).toBe(otherT5);
  });

  it('fills a slot the switch cleared as incompatible, and still reports the loss', () => {
    const model = withDefaults('flux', { t5_encoder: t5Encoder.key });
    // A Qwen VL encoder fits no FLUX slot, so a stale one is cleared; the T5 default is not what it held.
    const current = createSettings(createModel('qwen-image', { format: 'diffusers' }), {
      qwenVLEncoderModel: qwenVlEncoder,
    });
    const result = switchTo(model, [t5Encoder, qwenVlEncoder], current);

    expect(result.settings.qwenVLEncoderModel).toBeNull();
    expect(result.settings.t5EncoderModel).toBe(t5Encoder);
    expect(isLossyModelSwitch(result)).toBe(true);
  });

  it('skips unknown slot names, models that are not installed and models the slot rejects', () => {
    const model = withDefaults('flux', {
      clip_embed_model: t5Encoder.key,
      not_a_slot: clipEmbed.key,
      t5_encoder: 'uninstalled',
    });
    const { settings } = switchTo(model, [t5Encoder, clipEmbed]);

    expect(settings.t5EncoderModel).toBeNull();
    expect(settings.clipEmbedModel).toBeNull();
  });

  it('changes nothing when the model is selected again', () => {
    const model = withDefaults('flux', { t5_encoder: t5Encoder.key });
    const current = createSettings(model);

    expect(switchTo(model, [t5Encoder], current).settings.t5EncoderModel).toBeNull();
  });

  it('skips the FLUX.2 component source, which the family offers no slot for and picks automatically', () => {
    const source = createModel('flux2', { format: 'diffusers', key: 'klein-a', variant: 'klein_9b' });
    const model = createModel('flux2', {
      default_settings: { components: { component_source: source.key } },
      format: 'gguf_quantized',
      key: 'klein-default',
      variant: 'klein_9b',
    });

    expect(getModelDefaultComponents(model, createSettings(model), [source] as ComponentModelConfig[])).toEqual({});
  });

  it('opens a dependent slot once the slot it hangs on is filled, whatever order they are stored in', () => {
    const encoder = {
      base: 'any',
      format: 'gguf_quantized',
      key: 'qwen3-vl-8b-gguf',
      name: 'Qwen3-VL 8B',
      type: 'qwen3_vl_encoder',
      variant: 'qwen3_vl_8b',
    };
    const tower = { ...encoder, key: 'mmproj', name: 'mmproj', type: 'qwen3_vl_vision_tower' };
    const model = withDefaults('qwen-image-2-1', { qwen3_vl_vision_tower: tower.key, qwen3_vl_encoder: encoder.key });
    const current = createSettings(model);

    expect(getModelDefaultComponents(model, current, [tower, encoder] as ComponentModelConfig[])).toEqual({
      qwen3VLEncoderModel: encoder,
      qwen3VLVisionTowerModel: tower,
    });
    // Without the encoder default the tower has no slot to fill.
    expect(
      getModelDefaultComponents(withDefaults('qwen-image-2-1', { qwen3_vl_vision_tower: tower.key }), current, [
        tower,
      ] as ComponentModelConfig[])
    ).toEqual({});
    // The component list the Model Manager offers follows the same rule.
    const offered = (components: Record<string, string>) =>
      getDefaultComponentSlotPolicies(withDefaults('qwen-image-2-1', components), [
        encoder,
        tower,
      ] as ComponentModelConfig[]).map((slot) => slot.persistenceKey);

    expect(offered({})).not.toContain('qwen3_vl_vision_tower');
    expect(offered({ qwen3_vl_encoder: encoder.key })).toContain('qwen3_vl_vision_tower');
  });

  it('does not keep a selection that is no longer installed over the default of the new model', () => {
    const model = withDefaults('flux', { t5_encoder: t5Encoder.key });
    const removed = createSettings(createModel('flux', { key: 'flux-other' }), { t5EncoderModel: otherT5 });

    expect(switchTo(model, [t5Encoder], removed).settings.t5EncoderModel).toBe(t5Encoder);
  });

  it('clears a vision tower whose slot a reset to the default encoder closes', () => {
    const gguf = {
      base: 'any',
      format: 'gguf_quantized',
      key: 'encoder-gguf',
      name: 'GGUF encoder',
      type: 'qwen3_vl_encoder',
      variant: 'qwen3_vl_8b',
    };
    const checkpoint = { ...gguf, format: 'checkpoint', key: 'encoder-st', name: 'Checkpoint encoder' };
    const tower = { ...gguf, key: 'mmproj', name: 'mmproj', type: 'qwen3_vl_vision_tower' };
    const model = withDefaults('qwen-image-2-1', { qwen3_vl_encoder: checkpoint.key });
    const current = createSettings(model, {
      qwen3VLEncoderModel: gguf as ComponentModelConfig,
      qwen3VLVisionTowerModel: tower as ComponentModelConfig,
    });
    const models = [gguf, checkpoint, tower] as ComponentModelConfig[];

    expect(getModelDefaultComponents(model, current, models, { replace: true })).toEqual({
      qwen3VLEncoderModel: checkpoint,
      qwen3VLVisionTowerModel: null,
    });
    // A fitting selection that stays closes nothing.
    expect(getModelDefaultComponents(model, current, models)).toEqual({});
  });

  it('offers and fills no encoder default for a self-contained FLUX.1 SDNQ pipeline, whose graph loads its own', () => {
    const pipeline = { text_encoder: {}, text_encoder_2: {}, tokenizer: {}, tokenizer_2: {}, transformer: {}, vae: {} };
    const components = { clip_embed_model: clipEmbed.key, t5_encoder: t5Encoder.key };
    const sdnq = withDefaults('flux', components, { format: 'sdnq_quantized', submodels: pipeline } as never);
    const standalone = withDefaults('flux', components);
    const installed = [t5Encoder, clipEmbed] as ComponentModelConfig[];
    const offered = (model: GenerateModelConfig) =>
      getDefaultComponentSlotPolicies(model, installed).map((slot) => slot.persistenceKey);

    expect(offered(standalone)).toEqual(expect.arrayContaining(['t5_encoder', 'clip_embed_model']));
    expect(offered(sdnq)).not.toContain('t5_encoder');
    expect(offered(sdnq)).not.toContain('clip_embed_model');
    expect(getModelDefaultComponents(sdnq, createSettings(sdnq), installed)).toEqual({});
  });

  it.each([
    ['ideogram-4', 'ideogram4_unconditional_model'],
    ['wan', 'component_source'],
    ['qwen-image', 'component_source'],
  ])('offers and fills no %s default for the %s a Diffusers pipeline never loads', (base, persistenceKey) => {
    const main = (key: string, format: string) =>
      ({
        base,
        format,
        key,
        name: key,
        type: 'main',
        ...(base === 'wan' ? { variant: 't2v_a14b' } : {}),
      }) as MainModelConfig;
    const other = main('other', 'diffusers');
    const installed = [other] as ComponentModelConfig[];
    const offered = (model: GenerateModelConfig) =>
      getDefaultComponentSlotPolicies(model, installed).map((slot) => slot.persistenceKey);
    const diffusers = {
      ...main('pipeline', 'diffusers'),
      default_settings: { components: { [persistenceKey]: 'other' } },
    };
    const singleFile = { ...main('single-file', 'gguf_quantized'), default_settings: diffusers.default_settings };

    expect(offered(singleFile)).toContain(persistenceKey);
    expect(offered(diffusers)).not.toContain(persistenceKey);
    expect(getModelDefaultComponents(diffusers, createSettings(diffusers), installed)).toEqual({});
  });

  it('prunes a stored vision tower once the encoder it hangs on is gone, or no longer opens it', () => {
    const gguf = {
      base: 'any',
      format: 'gguf_quantized',
      key: 'encoder-gguf',
      name: 'GGUF encoder',
      type: 'qwen3_vl_encoder',
      variant: 'qwen3_vl_8b',
    };
    const checkpoint = { ...gguf, format: 'checkpoint', key: 'encoder-st', name: 'Checkpoint encoder' };
    const tower = { ...gguf, key: 'mmproj', name: 'mmproj', type: 'qwen3_vl_vision_tower' };
    const installed = [gguf, checkpoint, tower] as ComponentModelConfig[];
    const prune = (components: Record<string, string>) =>
      pruneClosedDefaultComponents(withDefaults('qwen-image-2-1', components), installed);

    expect(prune({ qwen3_vl_encoder: gguf.key, qwen3_vl_vision_tower: tower.key })).toEqual({
      qwen3_vl_encoder: gguf.key,
      qwen3_vl_vision_tower: tower.key,
    });
    // Cleared encoder; replaced by one with no tower slot.
    expect(prune({ qwen3_vl_vision_tower: tower.key })).toEqual({});
    expect(prune({ qwen3_vl_encoder: checkpoint.key, qwen3_vl_vision_tower: tower.key })).toEqual({
      qwen3_vl_encoder: checkpoint.key,
    });
    // An encoder that is no longer installed is the user's to see and clear first; keys of unknown slots stay.
    expect(prune({ qwen3_vl_encoder: 'deleted', qwen3_vl_vision_tower: tower.key, unknown_slot: 'x' })).toEqual({
      qwen3_vl_encoder: 'deleted',
      qwen3_vl_vision_tower: tower.key,
      unknown_slot: 'x',
    });
  });

  describe('dependent slots are judged against the final settings', () => {
    const gguf = {
      base: 'any',
      format: 'gguf_quantized',
      key: 'encoder-gguf',
      name: 'GGUF encoder',
      type: 'qwen3_vl_encoder',
      variant: 'qwen3_vl_8b',
    };
    const checkpoint = { ...gguf, format: 'checkpoint', key: 'encoder-st', name: 'Checkpoint encoder' };
    const tower = { ...gguf, key: 'mmproj', name: 'mmproj', type: 'qwen3_vl_vision_tower' };
    const installed = [gguf, checkpoint, tower] as ComponentModelConfig[];

    it.each([
      ['tower before encoder', { qwen3_vl_vision_tower: tower.key, qwen3_vl_encoder: checkpoint.key }],
      ['encoder before tower', { qwen3_vl_encoder: checkpoint.key, qwen3_vl_vision_tower: tower.key }],
    ])('leaves no hidden tower whatever the stored order (%s)', (_name, components) => {
      const model = withDefaults('qwen-image-2-1', components);
      const current = createSettings(model, {
        qwen3VLEncoderModel: gguf as ComponentModelConfig,
        qwen3VLVisionTowerModel: tower as ComponentModelConfig,
      });

      expect(getModelDefaultComponents(model, current, installed, { replace: true })).toEqual({
        qwen3VLEncoderModel: checkpoint,
        qwen3VLVisionTowerModel: null,
      });
    });
  });

  describe('the Wan low-noise expert', () => {
    const wan = (key: string, extra: Record<string, unknown> = {}) =>
      createModel('wan', { format: 'gguf_quantized', key, name: key, variant: 't2v_a14b', ...extra } as never);
    const high = wan('high');
    const low = wan('low');
    const offered = (model: GenerateModelConfig, models: unknown[]) =>
      getDefaultComponentSlotPolicies(model, models as ComponentModelConfig[]).find(
        (slot) => slot.persistenceKey === 'wan_transformer_low_noise'
      );

    it('accepts only a different single-file expert of the same variant for a single-file A14B model', () => {
      const slot = offered(high, [high, low])!;

      expect(slot.filter(low as never)).toBe(true);
      expect(slot.filter(wan('low-st', { format: 'checkpoint' }) as never)).toBe(true);
      expect(slot.filter(high as never)).toBe(false);
      expect(slot.filter(wan('pipeline', { format: 'diffusers' }) as never)).toBe(false);
      expect(slot.filter(wan('other-variant', { variant: 'i2v_a14b' }) as never)).toBe(false);
    });

    it("refuses two experts the loader's tags name for the same role, and accepts untagged ones", () => {
      const tagged = (key: string, expert: string) => wan(key, { expert });
      const slotOf = (main: GenerateModelConfig) => offered(main, [main])!;

      expect(slotOf(tagged('high-a', 'high')).filter(tagged('high-b', 'high') as never)).toBe(false);
      expect(slotOf(tagged('low-a', 'low')).filter(tagged('low-b', 'low') as never)).toBe(false);
      expect(slotOf(tagged('high-a', 'high')).filter(tagged('low-b', 'low') as never)).toBe(true);
      expect(slotOf(tagged('none-a', 'none')).filter(tagged('none-b', 'none') as never)).toBe(true);
      expect(slotOf(tagged('high-a', 'high')).filter(wan('untagged') as never)).toBe(true);
    });

    it('reports a saved expert the loader would reject, which the picker no longer lists', () => {
      const pipeline = wan('pipeline', { format: 'diffusers' });
      const reasons = (expert: unknown) =>
        getGenerationValidationReasons(
          high,
          createSettings(high, { componentSourceModel: pipeline, wanLowNoiseModel: expert as never })
        );

      expect(reasons(null)).toEqual([]);
      expect(reasons(low)).toEqual([]);
      expect(reasons(pipeline)).toEqual([
        'The selected low-noise expert is not a different single-file model of this variant. Select one or clear it.',
      ]);
    });

    it('lets a Diffusers component source supply the VAE only when its VAE width fits the model', () => {
      const a14bPipeline = wan('a14b-pipeline', { format: 'diffusers' });
      const smallPipeline = wan('small-pipeline', { format: 'diffusers', variant: 'ti2v_5b' });
      const wanVae = {
        base: 'wan',
        key: 'wan-vae-16',
        latent_channels: 16,
        name: 'Wan VAE',
        type: 'vae',
      } as VaeModelConfig;
      const reasons = (source: unknown, vae: VaeModelConfig | null = null) =>
        getGenerationValidationReasons(
          high,
          createSettings(high, { componentSourceModel: source as never, vae, wanT5EncoderModel: null })
        );

      expect(reasons(a14bPipeline)).toEqual([]);
      // The source still supplies the text encoder beside a VAE that fits.
      expect(reasons(smallPipeline, wanVae)).toEqual([]);
      expect(reasons(smallPipeline)).toEqual([
        'Generate needs a VAE for Wan models: a component source supplies one only when it is the same variant family.',
      ]);
    });

    it('is no default of a Diffusers or TI2V-5B model, whose loader ignores or reads the pair itself', () => {
      expect(offered(wan('pipeline', { format: 'diffusers' }), [low])).toBeUndefined();
      expect(offered(wan('small', { variant: 'ti2v_5b' }), [low])).toBeUndefined();
    });

    it('offers no slot to a model whose loader never reads it, and a switch clears a stale pick', () => {
      const pipeline = wan('pipeline', { format: 'diffusers' });
      const slotKeys = (model: GenerateModelConfig) =>
        getComponentSectionPolicy(model, createSettings(model)).slots.map((candidate) => candidate.key);

      expect(slotKeys(high)).toContain('wanLowNoiseModel');
      expect(slotKeys(pipeline)).not.toContain('wanLowNoiseModel');
      expect(slotKeys(wan('small', { variant: 'ti2v_5b' }))).not.toContain('wanLowNoiseModel');

      const current = createSettings(high, { wanLowNoiseModel: low });
      const result = getGenerateModelSelectionResult({
        currentValues: current,
        model: pipeline,
        models: [high, low, pipeline] as ComponentModelConfig[],
      });

      expect(result.settings.wanLowNoiseModel).toBeNull();
    });

    it('judges a kept expert by the catalog record, so a valid default replaces one reclassified since', () => {
      const model = withDefaults('wan', { wan_transformer_low_noise: low.key }, {
        format: 'gguf_quantized',
        key: high.key,
        variant: 't2v_a14b',
      } as never);
      // The project saved this pick while it was still a T2V expert; the catalog has since reclassified it.
      const old = wan('old');
      const reclassified = wan('old', { variant: 'i2v_a14b' });
      const saved = createSettings(model, { wanLowNoiseModel: old as never });

      expect(getModelDefaultComponents(model, saved, [model, low, reclassified] as never)).toEqual({
        wanLowNoiseModel: low,
      });
      // Still a T2V expert in the catalog: it stays.
      expect(getModelDefaultComponents(model, saved, [model, low, old] as never)).toEqual({});
    });

    it('skips a stored default the loader would reject', () => {
      const model = withDefaults('wan', { wan_transformer_low_noise: high.key }, {
        format: 'gguf_quantized',
        key: high.key,
        variant: 't2v_a14b',
      } as never);

      expect(getModelDefaultComponents(model, createSettings(model), [model, low] as never)).toEqual({});
    });
  });

  describe('the default VAE', () => {
    it('fills an empty VAE of a family with a VAE slot', () => {
      const model = createModel('flux', { default_settings: { vae: fluxVae.key }, format: 'checkpoint' });

      expect(switchTo(model, [fluxVae]).settings.vae).toBe(fluxVae);
    });

    it('fills the FLUX.2 VAE a single-file MOSS-Image takes as its default', () => {
      const model = createModel('moss-image', { default_settings: { vae: flux2Vae.key }, format: 'checkpoint' });

      expect(isComponentCompatibleWithModel(model, createSettings(model), 'vae', flux2Vae)).toBe(true);
      expect(switchTo(model, [flux2Vae]).settings.vae).toBe(flux2Vae);
      // SDXL's VAE is not among the ones MOSS-Image accepts.
      expect(
        switchTo(createModel('moss-image', { default_settings: { vae: sdxlVae.key }, format: 'checkpoint' }), [sdxlVae])
          .settings.vae
      ).toBeNull();
    });

    it('fills no VAE or Ministral default for a Diffusers ERNIE-Image, whose loader reads both from the pipeline', () => {
      const ministral = {
        base: 'any',
        key: 'ministral',
        name: 'Ministral 3B',
        type: 'mistral_encoder',
        variant: 'ministral3_3b',
      } as ComponentModelConfig;
      const ernie = (format: string) =>
        createModel('ernie-image', {
          default_settings: { components: { mistral_encoder: ministral.key }, vae: flux2Vae.key },
          format,
        });

      const single = switchTo(ernie('checkpoint'), [flux2Vae, ministral]).settings;
      const diffusers = switchTo(ernie('diffusers'), [flux2Vae, ministral]).settings;

      expect(single).toMatchObject({ mistralEncoderModel: ministral, vae: flux2Vae });
      expect(diffusers.mistralEncoderModel).toBeNull();
      expect(diffusers.vae).toBeNull();
    });

    it('replaces a VAE of another family, as left behind by a fallback from an unsupported model', () => {
      const model = createModel('flux', { default_settings: { vae: fluxVae.key }, format: 'checkpoint' });
      const stale = createSettings(createModel('sdxl'), { vae: sdxlVae });

      expect(getModelDefaultFills(model, stale, [fluxVae, sdxlVae]).vae).toBe(fluxVae);
    });

    it('judges a kept VAE by the catalog record, so a default replaces one reclassified since', () => {
      const defaultVae: VaeModelConfig = { base: 'flux', key: 'flux-vae-default', name: 'FLUX VAE 2', type: 'vae' };
      const model = createModel('flux', { default_settings: { vae: defaultVae.key }, format: 'checkpoint' });
      // The project saved the pick as a FLUX VAE; the catalog now classifies the same key as SD3's.
      const saved = createSettings(model, { vae: fluxVae });
      const reclassified: VaeModelConfig = { ...fluxVae, base: 'sd-3' };

      expect(getModelDefaultFills(model, saved, [defaultVae, reclassified]).vae).toBe(defaultVae);
      expect(getModelDefaultFills(model, saved, [defaultVae, fluxVae]).vae).toBeUndefined();
    });

    it('fills no VAE for a self-contained FLUX.1 SDNQ pipeline, whose graph decodes with its own', () => {
      const pipeline = {
        text_encoder: {},
        text_encoder_2: {},
        tokenizer: {},
        tokenizer_2: {},
        transformer: {},
        vae: {},
      };
      const model = createModel('flux', {
        default_settings: { vae: fluxVae.key },
        format: 'sdnq_quantized',
        submodels: pipeline,
      } as never);

      expect(switchTo(model, [fluxVae]).settings.vae).toBeNull();
    });

    it('keeps an installed VAE that fits when the model defaults to the bundled one', () => {
      const model = createModel('sdxl', { default_settings: { vae: 'default' } });
      const chosen = createSettings(createModel('sdxl', { key: 'sdxl-other' }), { vae: sdxlVae });

      expect(switchTo(model, [sdxlVae], chosen).settings.vae).toBe(sdxlVae);
    });

    it('clears a VAE that is gone when the model defaults to the bundled one', () => {
      const model = createModel('sdxl', { default_settings: { vae: 'default' } });
      const gone = createSettings(createModel('sdxl', { key: 'sdxl-other' }), { vae: sdxlVae });

      expect(switchTo(model, [], gone).settings.vae).toBeNull();
    });

    it('fills the SD VAE picker, and leaves a VAE the user chose', () => {
      const model = createModel('sdxl', { default_settings: { vae: sdxlVae.key } });
      const chosen: VaeModelConfig = { ...sdxlVae, key: 'chosen', name: 'Chosen' };
      const current = createSettings(createModel('sdxl', { key: 'sdxl-other' }), { vae: chosen });

      expect(switchTo(model, [sdxlVae]).settings.vae).toBe(sdxlVae);
      expect(switchTo(model, [sdxlVae, chosen], current).settings.vae).toBe(chosen);
    });

    it('leaves the bundled VAE for the "default" sentinel, and skips a VAE of another base', () => {
      expect(
        switchTo(createModel('sdxl', { default_settings: { vae: 'default' } }), [sdxlVae]).settings.vae
      ).toBeNull();
      expect(
        switchTo(createModel('sdxl', { default_settings: { vae: fluxVae.key } }), [fluxVae]).settings.vae
      ).toBeNull();
    });
  });
});

describe('getGuidanceSliderMax', () => {
  it('keeps the 10-wide track for floors below it', () => {
    expect(getGuidanceSliderMax(1, null)).toBe(10);
    expect(getGuidanceSliderMax(1, 20)).toBe(10);
    expect(getGuidanceSliderMax(1, 6)).toBe(6);
    expect(getGuidanceSliderMax(0, 100)).toBe(10);
  });

  it('spans a usable range above a floor at or beyond the track, so FLUX Fill runs from 25 to 35', () => {
    expect(getGuidanceSliderMax(25, null)).toBe(35);
    expect(getGuidanceSliderMax(10, null)).toBe(20);
  });

  it('never crosses the input ceiling or drops below the floor', () => {
    expect(getGuidanceSliderMax(25, 30)).toBe(30);
    expect(getGuidanceSliderMax(25, 25)).toBe(25);
  });
});
