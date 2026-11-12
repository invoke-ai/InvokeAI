/** Settings fixtures that satisfy every base's required components, shared by graph and recall coverage. */

import type {
  ComponentPolicyContext,
  ComponentSlotPolicy,
  GenerateComponentValueKey,
  ReferenceVisionComponents,
  SupportedGenerateBase,
} from './baseGenerationPolicies';
import type {
  ComponentModelConfig,
  ExternalImageGeneratorModelConfig,
  ExternalImageSize,
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateSettings,
  MainModelConfig,
  ModelIdentifierConfig,
} from './types';

import {
  GENERATE_COMPONENT_VALUE_KEYS,
  getComponentSectionPolicy,
  getDefaultGenerateSettings,
  getGenerationDimensions,
  getGenerationUiPolicy,
  getSchedulerOptions,
  isLoraSupported,
  isReferenceImageSupported,
  SUPPORTED_GENERATE_BASES,
  withAddedReferenceImages,
} from './baseGenerationPolicies';
import { getIsPidSupportedBase } from './pid';
import { clampDimension, deriveAspectRatioId } from './settings';

/** Compile both bundled and standalone component paths. */
export interface ModelShape {
  label: string;
  overrides: Partial<MainModelConfig>;
  /** Qwen-Image-2.1: read the references with a GGUF encoder and its Vision Tower instead of a safetensors encoder. */
  ggufEncoder?: true;
}

const DEFAULT_SHAPES: readonly ModelShape[] = [
  { label: 'diffusers', overrides: { format: 'diffusers' } },
  { label: 'standalone-components', overrides: { format: 'gguf_quantized' } },
];

/** Check per-base shape overrides against supported bases. */
export const SHAPE_OVERRIDES: Partial<Record<SupportedGenerateBase, readonly ModelShape[]>> = {
  // Anima ships single files only, checkpoint or GGUF; Anima-3.8B's variant adds a second encoder.
  anima: [
    { label: 'qwen3', overrides: { format: 'checkpoint', variant: 'anima_qwen3' } },
    { label: 'qwen3-gguf', overrides: { format: 'gguf_quantized', variant: 'anima_qwen3' } },
    { label: 'qwen35', overrides: { format: 'checkpoint', variant: 'anima_qwen35' } },
  ],
  // Ideogram's standalone fixture is a conditional checkpoint, not GGUF.
  'ideogram-4': [
    { label: 'diffusers', overrides: { format: 'diffusers' } },
    { label: 'standalone-components', overrides: { branch: 'conditional', format: 'checkpoint' } },
  ],
  // FLUX.2 dev and Klein need distinct encoder variants.
  flux2: [
    { label: 'dev-diffusers', overrides: { format: 'diffusers', variant: 'dev' } },
    { label: 'dev-standalone', overrides: { format: 'gguf_quantized', variant: 'dev' } },
    { label: 'klein-9b-standalone', overrides: { format: 'gguf_quantized', variant: 'klein_9b' } },
  ],
  // The Vision Tower slot opens only beside a GGUF encoder, so that path needs its own shape.
  'qwen-image-2-1': [
    ...DEFAULT_SHAPES,
    { ggufEncoder: true, label: 'standalone-gguf-encoder', overrides: { format: 'gguf_quantized' } },
  ],
  // FLUX.3 Klein samples without CFG; Klein Base compiles the negative branch at its CFG default.
  flux3: [
    { label: 'klein-diffusers', overrides: { format: 'diffusers', variant: 'flux3_klein' } },
    { label: 'klein-standalone', overrides: { format: 'gguf_quantized', variant: 'flux3_klein' } },
    { label: 'klein-base-diffusers', overrides: { format: 'diffusers', variant: 'flux3_klein_base' } },
  ],
};

const shapesForBase = (base: SupportedGenerateBase): readonly ModelShape[] => SHAPE_OVERRIDES[base] ?? DEFAULT_SHAPES;

/** Search component candidates through policy rather than copy compatibility rules. */
const CANDIDATE_BASES = ['any', ...SUPPORTED_GENERATE_BASES] as const;
const CANDIDATE_VARIANTS = [
  undefined,
  'qwen3_06b',
  'qwen3_4b',
  'qwen3_8b',
  'large',
  'gigantic',
  'dev',
  'klein_4b',
  'klein_9b',
  'ministral3_3b',
  'qwen3_vl_4b',
  'qwen3_vl_8b',
  'qwen3_5_4b',
  'qwen3_5_9b',
] as const;
/** Ideogram 4's two transformer branches; every other main leaves the field unset. */
const CANDIDATE_BRANCHES = [undefined, 'conditional', 'unconditional'] as const;
/** VAE widths. A served row can constrain the width as well as the base -- Wan ships 16 and 48. */
const CANDIDATE_LATENT_CHANNELS = [undefined, 16, 48] as const;

export const candidatesForSlot = (slot: ComponentSlotPolicy): ModelIdentifierConfig[] => {
  const candidates: ModelIdentifierConfig[] = [];

  for (const type of slot.modelTypes) {
    for (const base of CANDIDATE_BASES) {
      for (const variant of CANDIDATE_VARIANTS) {
        for (const latentChannels of type === 'vae' ? CANDIDATE_LATENT_CHANNELS : [undefined]) {
          for (const branch of type === 'main' ? CANDIDATE_BRANCHES : [undefined]) {
            candidates.push({
              base,
              // Distinguish bundled component sources from single-file branch formats.
              format: slot.valueKind === 'main' ? (branch ? 'checkpoint' : 'diffusers') : undefined,
              key: `${base}-${type}-${variant ?? 'novariant'}${latentChannels ? `-${latentChannels}` : ''}${branch ? `-${branch}` : ''}`,
              name: `${base} ${type} ${variant ?? ''} ${branch ?? ''}`.trim(),
              type,
              variant: variant ?? null,
              ...(branch ? { branch } : {}),
              ...(type === 'text_llm' ? { supports_images: true } : {}),
              ...(latentChannels ? { latent_channels: latentChannels } : {}),
            });
          }
        }
      }
    }
  }

  return candidates;
};

export const buildContext = (
  model: MainModelConfig,
  settings: GenerateSettings,
  slots: readonly ComponentSlotPolicy[]
): ComponentPolicyContext => {
  // Derive keys from slots so new slots enter coverage automatically.
  const keys = new Set<GenerateComponentValueKey>(slots.map((slot) => slot.key));
  const selectedComponents = {} as ComponentPolicyContext['selectedComponents'];

  for (const key of keys) {
    selectedComponents[key] = settings[key] as never;
  }

  return { model, settings, selectedComponents };
};

/** Fill to a fixed point: selecting a component source can change required slots. */
const satisfyRequiredComponents = (
  model: MainModelConfig,
  initial: GenerateSettings
): { settings: GenerateSettings; filled: GenerateComponentValueKey[] } => {
  let settings = initial;
  const filled: GenerateComponentValueKey[] = [];

  for (let pass = 0; pass < 5; pass++) {
    const { slots } = getComponentSectionPolicy(model, settings);
    const context = buildContext(model, settings, slots);
    let changed = false;

    for (const slot of slots) {
      if (!slot.required?.(context) || settings[slot.key]) {
        continue;
      }

      const candidate = candidatesForSlot(slot).find((c) => !slot.filter || slot.filter(c, context));

      if (candidate) {
        settings = { ...settings, [slot.key]: candidate };
        filled.push(slot.key);
        changed = true;
      }
    }

    if (!changed) {
      break;
    }
  }

  return { filled: filled.sort(), settings };
};

const createModel = (base: SupportedGenerateBase, shape: ModelShape): MainModelConfig => ({
  base,
  key: `${base}-${shape.label}`,
  name: `${base} (${shape.label})`,
  type: 'main',
  ...shape.overrides,
});

export const satisfiedSettingsFor = (base: SupportedGenerateBase, shape: ModelShape) => {
  const model = createModel(base, shape);
  const required = satisfyRequiredComponents(model, {
    ...getDefaultGenerateSettings(model),
    positivePrompt: 'a test prompt',
    seed: 1,
    seedMode: 'fixed',
  });

  if (!shape.ggufEncoder || !required.settings.qwen3VLEncoderModel) {
    return { model, ...required };
  }

  // The tower is optional, so it stays out of `filled` (the required components): pick the GGUF encoder, then what its slot offers.
  const settings = {
    ...required.settings,
    qwen3VLEncoderModel: { ...required.settings.qwen3VLEncoderModel, format: 'gguf_quantized' },
  };
  const slot = getComponentSectionPolicy(model, settings).slots.find(
    (candidate) => candidate.key === 'qwen3VLVisionTowerModel'
  );
  const tower = slot && candidatesForSlot(slot).find((c) => slot.filter?.(c, buildContext(model, settings, [slot])));

  return tower
    ? {
        model,
        filled: required.filled,
        settings: { ...settings, qwen3VLVisionTowerModel: tower },
      }
    : { model, ...required, settings };
};

/** Every supported base in each of its component shapes. */
export const generateGraphCases = SUPPORTED_GENERATE_BASES.flatMap((base) =>
  shapesForBase(base).map((shape) => ({ base, label: `${base} / ${shape.label}`, shape }))
);

/** Required components plus every optional slot a compatible candidate fits, with PiD decoding on where supported. */
export const fullyFilledSettingsFor = (base: SupportedGenerateBase, shape: ModelShape) => {
  const { filled, model, settings: required } = satisfiedSettingsFor(base, shape);
  let settings: GenerateSettings = { ...required, pidMode: getIsPidSupportedBase(base) ? 'fit' : 'off' };
  const allFilled = new Set(filled);

  // Fill to a fixed point: a filled slot can reveal or constrain another.
  for (let pass = 0; pass < 5; pass++) {
    const { slots } = getComponentSectionPolicy(model, settings);
    const context = buildContext(model, settings, slots);
    let changed = false;

    for (const slot of slots) {
      if (settings[slot.key]) {
        continue;
      }

      const candidate = candidatesForSlot(slot).find((c) => !slot.filter || slot.filter(c, context));

      if (candidate) {
        settings = { ...settings, [slot.key]: candidate };
        allFilled.add(slot.key);
        changed = true;
      }
    }

    if (!changed) {
      break;
    }
  }

  return { filled: [...allFilled].sort(), model, settings };
};

/** References for the families whose builders wire them through their own nodes rather than an adapter. */
const tunedReferenceImagesFor = (
  model: MainModelConfig,
  image: GenerateReferenceImageAsset,
  components: ReferenceVisionComponents
): GenerateReferenceImage[] | undefined => {
  if (model.base === 'krea-2') {
    return [{ config: { image, styleStrength: 1, type: 'krea2_reference_image' }, id: 'style', isEnabled: true }];
  }

  // Only an encoder with a vision tower reads references: a Diffusers pipeline's own, or a standalone safetensors
  // one; other shapes compile without them.
  if (model.base === 'qwen-image-2-1' && isReferenceImageSupported(model, components)) {
    return [{ config: { image, type: 'qwen_image_2_1_reference_image' }, id: 'edit', isEnabled: true }];
  }

  return undefined;
};

/**
 * The optional per-family tuning and output filters, all switched on, so the backend contract also covers the nodes
 * and literal values a default recipe never emits. Settings a family ignores are harmless to carry.
 */
export const tunedSettingsFor = (base: SupportedGenerateBase, shape: ModelShape) => {
  const { filled, model, settings } = satisfiedSettingsFor(base, shape);
  const image = { original: { image: { height: 1024, image_name: 'style-reference.png', width: 1024 } } };
  // Only a graph with the FLUX.3 prompt upsampler switched on contains its node, so switch it on and pick a model.
  const upsamplerOn = { ...settings, flux3PromptUpsamplerEnabled: base === 'flux3' };
  const upsamplerSlot = getComponentSectionPolicy(model, upsamplerOn).slots.find(
    (slot) => slot.key === 'flux3PromptUpsamplerModel'
  );
  const upsamplerModel =
    upsamplerSlot &&
    candidatesForSlot(upsamplerSlot).find((c) =>
      upsamplerSlot.filter?.(c, buildContext(model, upsamplerOn, [upsamplerSlot]))
    );

  return {
    filled,
    model,
    projectSettings: { useCpuNoise: true, useNsfwChecker: true, useWatermark: true },
    settings: {
      ...upsamplerOn,
      flux3PromptUpsamplerModel: upsamplerModel ?? null,
      // CFG above 1 adds the negative prompt, which Qwen-Image-2.1 also encodes with the references.
      cfgScale: base === 'qwen-image-2-1' ? 4 : settings.cfgScale,
      // Exercise explicit zero overrides as well as the default automatic metadata contract.
      hiDiffusionEnabled: base === 'sd-1' || base === 'sdxl',
      hiDiffusionT1Ratio: 0,
      hiDiffusionT2Ratio: 0,
      fluxDypeExponent: 1,
      fluxDypePreset: 'manual' as const,
      fluxDypeScale: 3,
      qwenImageEncoderQuantization: 'nf4' as const,
      qwenImageShift: 3,
      referenceImages: tunedReferenceImagesFor(model, image, settings) ?? settings.referenceImages,
      zImageSeedVarianceEnabled: true,
      zImageSeedVarianceStrength: 0.2,
      zImageShift: 3,
    },
  };
};

const RECALL_REFERENCE_IMAGE: GenerateReferenceImageAsset = {
  original: { image: { height: 512, image_name: 'recall-reference.png', width: 512 } },
};

/**
 * Every setting image recall restores, at a value the policy accepts and no default holds, so a round trip that drops
 * one shows. Values a family does not record are harmless to carry. Seamless tiling and CFG rescale stay at their
 * defaults where the panel hides them: recall restores them on every family.
 */
const withRecallTuning = (
  model: GenerateModelConfig,
  settings: GenerateSettings,
  size: ExternalImageSize
): GenerateSettings => {
  const defaults = getDefaultGenerateSettings(model);
  const policy = getGenerationUiPolicy(model);

  return {
    ...settings,
    // Canvas inpainting then runs without the adapter in the filled slot, so its record shows the switch off.
    animaInpaintAdapterEnabled: false,
    aspectRatioId: deriveAspectRatioId(size.width, size.height),
    aspectRatioValue: size.width / size.height,
    cfgRescaleMultiplier: policy.cfgRescaleVisible ? 0.3 : defaults.cfgRescaleMultiplier,
    // FLUX.3 Klein samples without CFG: validation refuses any scale but its default.
    cfgScale:
      model.base === 'flux3' && model.variant === 'flux3_klein'
        ? defaults.cfgScale
        : Math.min(policy.guidanceMax ?? Number.POSITIVE_INFINITY, defaults.cfgScale + 0.5),
    clipSkip: policy.clipSkipMax === null ? defaults.clipSkip : 2,
    fluxDypeExponent: 1,
    fluxDypePreset: 'manual',
    fluxDypeScale: 3,
    height: size.height,
    ...(policy.hiDiffusionVisible
      ? {
          hiDiffusionEnabled: true,
          hiDiffusionRauNetEnabled: false,
          hiDiffusionT1Ratio: 0.6,
          hiDiffusionT2Ratio: 0.2,
          hiDiffusionWindowAttentionEnabled: false,
        }
      : {}),
    ideogram4ColorPalette: ['teal', 'amber'],
    ideogram4GuidanceScale: 5,
    ideogram4Mu: 1,
    ideogram4SamplerPreset: 'V4_TURBO_12',
    ideogram4Steps: 30,
    krea2RebalanceEnabled: true,
    krea2RebalanceMultiplier: 3,
    krea2RebalanceWeights: '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,4.0,1.0,3.0,1.0',
    krea2SeedVarianceEnabled: true,
    krea2SeedVarianceRandomizePercent: 40,
    krea2SeedVarianceStrength: 0.3,
    loras: isLoraSupported(model)
      ? [
          {
            isEnabled: true,
            model: { base: model.base, key: `${model.base}-recall-lora`, name: `${model.base} LoRA`, type: 'lora' },
            weight: 0.75,
          },
        ]
      : [],
    mossImageUncondSkipLayers: [10, 22],
    negativePrompt: 'a recall-tuned negative prompt',
    pidSteps: 6,
    qwenImageEncoderQuantization: 'nf4',
    qwenImageShift: 2,
    scheduler: getSchedulerOptions(model)[1]!.value,
    seamlessXAxis: policy.seamlessVisible,
    seamlessYAxis: policy.seamlessVisible,
    steps: defaults.steps + 1,
    wanGuidanceScaleLowNoise: 3.5,
    width: size.width,
    zImageSeedVarianceEnabled: true,
    zImageSeedVarianceRandomizePercent: 40,
    zImageSeedVarianceStrength: 0.3,
    zImageShift: 2,
  };
};

/** The reference the family reads: an adapter where the base takes one, otherwise the base's own reference type. */
const recallReferenceImagesFor = (model: MainModelConfig, settings: GenerateSettings): GenerateReferenceImage[] => {
  if (model.base === 'krea-2') {
    return [
      {
        config: { image: RECALL_REFERENCE_IMAGE, styleStrength: 1.5, type: 'krea2_reference_image' },
        id: 'style',
        isEnabled: true,
      },
    ];
  }

  return isReferenceImageSupported(model, settings)
    ? withAddedReferenceImages(
        model,
        [{ base: model.base, key: `${model.base}-ip-adapter`, name: `${model.base} IP Adapter`, type: 'ip_adapter' }],
        [],
        [{ id: 'reference', image: RECALL_REFERENCE_IMAGE }]
      )
    : [];
};

/**
 * A shape with every component filled and every recalled setting tuned off its default, PiD decoding included, and
 * the family's reference where it reads one. The size is off the model's default and not square. Z-Image decodes in
 * native PiD, which generates at a quarter of the recorded size, and the other PiD families in fit.
 */
export const recallTunedSettingsFor = (base: SupportedGenerateBase, shape: ModelShape) => {
  const { model, settings: fullyFilled } = fullyFilledSettingsFor(base, shape);
  const filled: GenerateSettings =
    base === 'z-image' && fullyFilled.pidMode === 'fit' ? { ...fullyFilled, pidMode: 'native' } : fullyFilled;
  const defaults = getDefaultGenerateSettings(model);
  const { grid } = getGenerationDimensions(model, filled.pidMode);
  const settings = withRecallTuning(model, filled, {
    height: clampDimension(defaults.height * 0.75, grid),
    width: clampDimension(defaults.width + grid, grid),
  });

  return {
    label: `${base} / ${shape.label}`,
    model,
    settings: { ...settings, referenceImages: recallReferenceImagesFor(model, settings) },
  };
};

/** The installed models a recall resolves the record against, the way the gallery hands them over. */
export const recallCatalogFor = (model: GenerateModelConfig, settings: GenerateSettings) => ({
  models: [
    model,
    ...GENERATE_COMPONENT_VALUE_KEYS.flatMap((key): ComponentModelConfig[] => {
      const component = settings[key];

      return component ? [component] : [];
    }),
    ...settings.loras.map((lora) => lora.model),
    ...(settings.refinerModel ? [settings.refinerModel] : []),
    ...settings.referenceImages.flatMap(({ config }): ComponentModelConfig[] =>
      'model' in config && config.model ? [config.model] : []
    ),
  ],
  supportedModels: [model],
  vaeModels: settings.vae ? [settings.vae] : [],
});

/** SDXL finishing with its refiner, tuned off every refiner default; PiD, which replaces the decode it feeds, is off. */
export const sdxlRefinerRecallCase = () => {
  const { model, settings } = recallTunedSettingsFor('sdxl', {
    label: 'diffusers-with-refiner',
    overrides: { format: 'diffusers' },
  });
  const refiner: MainModelConfig = { base: 'sdxl-refiner', key: 'sdxl-refiner', name: 'SDXL Refiner', type: 'main' };

  return {
    label: 'sdxl / diffusers-with-refiner',
    model,
    settings: {
      ...settings,
      pidMode: 'off' as const,
      refinerCfgScale: 6,
      refinerModel: refiner,
      refinerNegativeAestheticScore: 2,
      refinerPositiveAestheticScore: 7,
      refinerScheduler: 'dpmpp_2m',
      refinerStart: 0.7,
      refinerSteps: 21,
    },
  };
};

/** FLUX with a Redux reference in place of the IP-Adapter the tuned recipe carries. */
export const fluxReduxRecallCase = () => {
  const { model, settings } = recallTunedSettingsFor('flux', {
    label: 'diffusers-with-redux',
    overrides: { format: 'diffusers' },
  });
  const redux: ComponentModelConfig = { base: 'flux', key: 'flux-redux', name: 'FLUX Redux', type: 'flux_redux' };

  return {
    label: 'flux / diffusers-with-redux',
    model,
    settings: {
      ...settings,
      referenceImages: [
        {
          config: {
            image: RECALL_REFERENCE_IMAGE,
            imageInfluence: 'high' as const,
            model: redux,
            type: 'flux_redux' as const,
          },
          id: 'redux',
          isEnabled: true,
        },
      ],
    },
  };
};

/** Krea-2 edit references in place of the style reference, read by the tuned LoRA, with the reference K/V cache on. */
export const krea2EditRecallCase = () => {
  const { model, settings } = recallTunedSettingsFor('krea-2', {
    label: 'diffusers-with-edit-references',
    overrides: { format: 'diffusers' },
  });

  return {
    label: 'krea-2 / diffusers-with-edit-references',
    model,
    settings: {
      ...settings,
      krea2ReferenceKvCache: true,
      referenceImages: ['edit-1', 'edit-2'].map((id) => ({
        config: { image: RECALL_REFERENCE_IMAGE, type: 'krea2_edit_reference_image' as const },
        id,
        isEnabled: true,
      })),
    },
  };
};

const externalRecallCase = (
  model: ExternalImageGeneratorModelConfig,
  size: ExternalImageSize,
  options: Partial<GenerateSettings>
) => ({
  label: `external / ${model.provider_id}`,
  model,
  settings: {
    ...withRecallTuning(
      model,
      { ...getDefaultGenerateSettings(model), positivePrompt: 'a test prompt', seed: 1, seedMode: 'fixed' },
      size
    ),
    ...options,
  },
});

/**
 * One model per hosted provider, each with its own options: GPT Image with a reference image, Gemini at one of its
 * named sizes, Seedream.
 */
export const externalRecallCases = () => [
  externalRecallCase(
    {
      base: 'external',
      capabilities: { max_reference_images: 4, modes: ['txt2img', 'img2img'], supports_reference_images: true },
      format: 'external_api',
      key: 'openai-gpt-image-1',
      name: 'GPT Image 1',
      provider_id: 'openai',
      provider_model_id: 'gpt-image-1',
      type: 'external_image_generator',
    },
    { height: 1536, width: 1152 },
    {
      openaiBackground: 'transparent',
      openaiInputFidelity: 'high',
      openaiQuality: 'high',
      referenceImages: [
        {
          config: { image: RECALL_REFERENCE_IMAGE, type: 'external_reference_image' },
          id: 'reference',
          isEnabled: true,
        },
      ],
    }
  ),
  externalRecallCase(
    {
      base: 'external',
      capabilities: {
        modes: ['txt2img'],
        resolution_presets: [
          { aspect_ratio: '1:1', height: 1024, image_size: '1K', label: 'Square', width: 1024 },
          { aspect_ratio: '16:9', height: 1152, image_size: '2K', label: 'Landscape', width: 2048 },
        ],
      },
      format: 'external_api',
      key: 'gemini-image',
      name: 'Gemini Image',
      provider_id: 'gemini',
      provider_model_id: 'gemini-3-pro-image-preview',
      type: 'external_image_generator',
    },
    { height: 1152, width: 2048 },
    { externalImageSize: '2K', geminiTemperature: 0.7, geminiThinkingLevel: 'high' }
  ),
  externalRecallCase(
    {
      base: 'external',
      capabilities: { modes: ['txt2img'], supports_seed: true },
      format: 'external_api',
      key: 'seedream-4-5',
      name: 'Seedream 4.5',
      provider_id: 'seedream',
      provider_model_id: 'seedream-4-5-251128',
      type: 'external_image_generator',
    },
    { height: 1536, width: 2048 },
    { seedreamOptimizePrompt: true, seedreamWatermark: true }
  ),
];
