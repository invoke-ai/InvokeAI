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
  addReferenceImages,
  GENERATE_COMPONENT_VALUE_KEYS,
  getComponentSectionPolicy,
  getDefaultGenerateSettings,
  getGenerationDimensions,
  getGenerationUiPolicy,
  getSchedulerOptions,
  isLoraSupported,
  isReferenceImageSupported,
  SUPPORTED_GENERATE_BASES,
} from './baseGenerationPolicies';
import { getIsPidSupportedBase } from './pid';
import { clampDimension, deriveAspectRatioId } from './settings';

export interface ModelShape {
  label: string;
  overrides: Partial<MainModelConfig>;
  /** Read references through a GGUF encoder and its Vision Tower instead of a safetensors encoder. */
  ggufEncoder?: true;
}

const DEFAULT_SHAPES: readonly ModelShape[] = [
  { label: 'diffusers', overrides: { format: 'diffusers' } },
  { label: 'standalone-components', overrides: { format: 'gguf_quantized' } },
];

const SHAPE_OVERRIDES: Partial<Record<SupportedGenerateBase, readonly ModelShape[]>> = {
  anima: [
    { label: 'qwen3', overrides: { format: 'checkpoint', variant: 'anima_qwen3' } },
    { label: 'qwen3-gguf', overrides: { format: 'gguf_quantized', variant: 'anima_qwen3' } },
    { label: 'qwen35', overrides: { format: 'checkpoint', variant: 'anima_qwen35' } },
  ],
  'ideogram-4': [
    { label: 'diffusers', overrides: { format: 'diffusers' } },
    { label: 'standalone-components', overrides: { branch: 'conditional', format: 'checkpoint' } },
  ],
  // FLUX.1 loads from a single file or a quantized transformer; Generate refuses a plain Diffusers folder.
  flux: [
    { label: 'checkpoint', overrides: { format: 'checkpoint' } },
    { label: 'standalone-components', overrides: { format: 'gguf_quantized' } },
  ],
  flux2: [
    { label: 'dev-diffusers', overrides: { format: 'diffusers', variant: 'dev' } },
    { label: 'dev-standalone', overrides: { format: 'gguf_quantized', variant: 'dev' } },
    { label: 'klein-9b-standalone', overrides: { format: 'gguf_quantized', variant: 'klein_9b' } },
  ],
  // The Vision Tower slot opens only beside a GGUF encoder, so that path needs its own shape (Krea-2 below).
  'qwen-image-2-1': [
    ...DEFAULT_SHAPES,
    { ggufEncoder: true, label: 'standalone-gguf-encoder', overrides: { format: 'gguf_quantized' } },
  ],
  'krea-2': [
    ...DEFAULT_SHAPES,
    { ggufEncoder: true, label: 'standalone-gguf-encoder', overrides: { format: 'gguf_quantized' } },
  ],
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

  // The tower is optional, so it stays out of `filled`.
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

export const generateGraphCases = SUPPORTED_GENERATE_BASES.flatMap((base) =>
  shapesForBase(base).map((shape) => ({ base, label: `${base} / ${shape.label}`, shape }))
);

export const fullyFilledSettingsFor = (base: SupportedGenerateBase, shape: ModelShape) => {
  const { filled, model, settings: required } = satisfiedSettingsFor(base, shape);
  let settings: GenerateSettings = { ...required, pidMode: getIsPidSupportedBase(base) ? 'fit' : 'off' };
  const allFilled = new Set(filled);

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

const tunedReferenceImagesFor = (
  model: MainModelConfig,
  image: GenerateReferenceImageAsset,
  components: ReferenceVisionComponents
): GenerateReferenceImage[] | undefined => {
  if (model.base === 'krea-2') {
    return [{ config: { image, styleStrength: 1, type: 'krea2_reference_image' }, id: 'style', isEnabled: true }];
  }

  if (model.base === 'qwen-image-2-1' && isReferenceImageSupported(model, components)) {
    return [{ config: { image, type: 'qwen_image_2_1_reference_image' }, id: 'edit', isEnabled: true }];
  }

  return undefined;
};

/** Optional tuning and filters all on, so the backend contract covers nodes a default recipe never emits. */
export const tunedSettingsFor = (base: SupportedGenerateBase, shape: ModelShape) => {
  const { filled, model, settings } = satisfiedSettingsFor(base, shape);
  const image = { original: { image: { height: 1024, image_name: 'style-reference.png', width: 1024 } } };
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
      cfgScale: base === 'qwen-image-2-1' ? 4 : settings.cfgScale,
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

/** Every recalled setting off its default, so a round trip that drops one shows. */
const withRecallTuning = (
  model: GenerateModelConfig,
  settings: GenerateSettings,
  size: ExternalImageSize
): GenerateSettings => {
  const defaults = getDefaultGenerateSettings(model);
  const policy = getGenerationUiPolicy(model);

  return {
    ...settings,
    animaInpaintAdapterEnabled: false,
    aspectRatioId: deriveAspectRatioId(size.width, size.height),
    aspectRatioValue: size.width / size.height,
    cfgRescaleMultiplier: policy.cfgRescaleVisible ? 0.3 : defaults.cfgRescaleMultiplier,
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
    ideogram4Steps: 31,
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

  return addReferenceImages({
    additions: [{ id: 'reference', image: RECALL_REFERENCE_IMAGE }],
    components: settings,
    mode: 'strict',
    model,
    models: [
      { base: model.base, key: `${model.base}-ip-adapter`, name: `${model.base} IP Adapter`, type: 'ip_adapter' },
    ],
    referenceImages: [],
  }).referenceImages;
};

/** Everything filled and tuned; the size is off default and not square (Z-Image PiD is native, others fit). */
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

export const fluxReduxRecallCase = () => {
  const { model, settings } = recallTunedSettingsFor('flux', {
    label: 'checkpoint-with-redux',
    overrides: { format: 'checkpoint' },
  });
  const redux: ComponentModelConfig = { base: 'flux', key: 'flux-redux', name: 'FLUX Redux', type: 'flux_redux' };

  return {
    label: 'flux / checkpoint-with-redux',
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
