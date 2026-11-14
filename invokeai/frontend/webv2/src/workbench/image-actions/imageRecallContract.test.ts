/**
 * The image recall contract. Every key an image graph records, and every key the backend declares for
 * `core_metadata`, has an entry here; the gates hold each entry to what recall does with records compiled from
 * settings tuned off every default. A key a builder starts to write fails G1 until it has an entry, and a key the
 * backend starts to declare fails `tsc` until it has one.
 */

import type { components } from '@api/schema';
import type { GalleryImage } from '@features/gallery';
import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateSettings,
  GenerateWidgetValues,
  MainModelConfig,
} from '@features/generation/contracts';
import type { CanvasCompileMode } from '@features/generation/core/canvas/types';
import type { SpandrelModelConfig } from '@features/upscale/core/types';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { canvasModesFor, compile as compileCanvas } from '@features/generation/core/canvas/compileCanvasGraph.testing';
import {
  externalRecallCases,
  fluxReduxRecallCase,
  generateGraphCases,
  krea2EditRecallCase,
  recallCatalogFor,
  recallTunedSettingsFor,
  sdxlRefinerRecallCase,
} from '@features/generation/core/graphCoverage.testing';
import { compileGenerateGraph } from '@features/generation/graph';
import {
  deriveAspectRatioId,
  getDefaultGenerateSettings,
  getExternalResolutionPresets,
  getGenerationUiPolicy,
  isSupportedGenerateModel,
  isVaeModelConfig,
} from '@features/generation/settings';
import { compileModelOnlyUpscaleGraph, compileUpscaleGraph } from '@features/upscale/core/graph';
import { CASES as UPSCALE_CASES, INSTALLED, valuesForCase } from '@features/upscale/core/upscaleGraphCoverage.testing';
import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { describe, expect, expectTypeOf, it } from 'vitest';

import {
  buildImageRecallSettings,
  getImageRecallCapabilities,
  type ImageRecallKind,
  type ImageRecallResult,
} from './imageRecall';
import { isRuntimeValue, type KnownKeys, recordOf } from './recallContract.testing';

type Written = Readonly<Record<string, unknown>>;
type Setting = keyof GenerateWidgetValues;

/** A key recall applies to the Generate settings. */
interface Restores {
  restores: readonly Setting[];
  /** The recall kinds that apply it; Recall All and Remix unless listed. */
  kinds?: readonly ImageRecallKind[];
  /** False where recall leaves the key alone on purpose; the ablation (G3) then proves it does. */
  when?: (model: GenerateModelConfig, written: Written) => boolean;
  /** What the settings hold after recall, where that is not what the record was compiled from. */
  expected?: (written: Written, tuned: GenerateWidgetValues) => Partial<GenerateWidgetValues>;
  /** What recall sets when a generation record of the model's family lacks the key (G8); null where it sets nothing. */
  absent?: (model: GenerateModelConfig, written: Written) => Partial<GenerateWidgetValues> | null;
}

type Entry =
  | Restores
  /** Read to interpret the record, or by another recall path, never into one Generate setting of its own. */
  | { consumed: string }
  /** Not restored, on purpose. `declaredOnly`: the backend declares it, no image graph writes it. */
  | { ignored: string; declaredOnly?: true };

type DeclaredCoreMetadataKey = Exclude<
  KnownKeys<components['schemas']['CoreMetadataInvocation']>,
  'id' | 'is_intermediate' | 'type' | 'use_cache'
>;

const VIDEO_PROFILE = 'Video profile: covered by videoRecallContract.test.ts.';
const RUNTIME_VALUE = 'A node computes it from the prompt while the graph runs; recall restores the prompt itself.';
const UPSCALE_SETTING = 'An Upscale panel setting; image recall restores the Generate panel only.';

/** A Diffusers main records itself as its own component source; having no source slot, recall skips it. */
const unlessMainModel = (key: string) => (model: GenerateModelConfig, written: Written) =>
  (written[key] as { key?: unknown } | null)?.key !== model.key;

const ANIMA_ADAPTER_MODES: readonly unknown[] = ['anima_inpaint', 'anima_outpaint'];

const IMAGE_RECALL_CONTRACT = {
  // What the record is.
  generation_mode: {
    consumed:
      'Marks a generation record, on which a key left out ran at its default (G8); an Anima inpaint or outpaint ' +
      'mode without `anima_inpaint_adapter` switches the adapter off.',
  },
  model: { restores: ['model', 'modelKey'] },
  rand_device: { ignored: 'Follows the project noise setting (`useCpuNoise`), which Generate does not hold.' },
  app_version: { ignored: 'Stamped by the backend; a model-only upscale copies it from its source.' },
  metadata_version: { ignored: 'Stamped by the backend; a model-only upscale copies it from its source.' },

  // Prompts, seed and size.
  positive_prompt: {
    kinds: ['all', 'remix', 'prompts'],
    restores: ['positivePrompt', 'promptTemplate', 'promptTemplateViewMode'],
  },
  negative_prompt: {
    kinds: ['all', 'remix', 'prompts'],
    restores: ['negativePrompt', 'promptTemplate', 'promptTemplateViewMode'],
  },
  seed: { kinds: ['all', 'seed'], restores: ['seed', 'seedMode'] },
  // An upscale records the size of the frame it enlarged, which only exists once the graph runs.
  width: {
    restores: ['width', 'aspectRatioId', 'aspectRatioValue'],
    when: (_model, written) => typeof written.width === 'number',
  },
  height: {
    restores: ['height', 'aspectRatioId', 'aspectRatioValue'],
    when: (_model, written) => typeof written.height === 'number',
  },

  // Sampling. Recall restores CFG rescale and seamless tiling on every family; the fixtures carry their defaults
  // where the panel hides them.
  steps: { restores: ['steps'] },
  cfg_scale: { restores: ['cfgScale'] },
  guidance: { restores: ['cfgScale'], when: (model) => getGenerationUiPolicy(model).guidanceLabel === 'Guidance' },
  cfg_rescale_multiplier: {
    restores: ['cfgRescaleMultiplier'],
    when: (model) => getGenerationUiPolicy(model).cfgRescaleVisible,
  },
  // A family without a scheduler choice records its fixed one, which recall restores as recorded.
  scheduler: { expected: (written) => ({ scheduler: String(written.scheduler) }), restores: ['scheduler'] },
  seamless_x: { restores: ['seamlessXAxis'], when: (model) => getGenerationUiPolicy(model).seamlessVisible },
  seamless_y: { restores: ['seamlessYAxis'], when: (model) => getGenerationUiPolicy(model).seamlessVisible },
  clip_skip: {
    kinds: ['all', 'remix', 'clipSkip'],
    restores: ['clipSkip'],
    when: (model) => getGenerationUiPolicy(model).clipSkipMax !== null,
  },
  hidiffusion: { restores: ['hiDiffusionEnabled'] },
  hidiffusion_raunet: { restores: ['hiDiffusionRauNetEnabled'] },
  hidiffusion_window_attn: { restores: ['hiDiffusionWindowAttentionEnabled'] },
  hidiffusion_t1_ratio: { restores: ['hiDiffusionT1Ratio'] },
  hidiffusion_t2_ratio: { restores: ['hiDiffusionT2Ratio'] },
  hidiffusion_auto_ratios: {
    consumed: 'Says whether the recorded ratios are automatic, which restores them as unset.',
  },

  // Components.
  vae: { restores: ['vae'] },
  qwen_image_vae: { consumed: "Qwen-Image's VAE alias, read only from records without `vae`; the graph writes both." },
  clip_embed_model: { restores: ['clipEmbedModel'] },
  t5_encoder: { restores: ['t5EncoderModel'] },
  mistral_encoder: { restores: ['mistralEncoderModel'] },
  qwen3_encoder: { restores: ['qwen3EncoderModel'] },
  qwen3_5_encoder: { restores: ['qwen35EncoderModel'] },
  qwen3_vl_encoder: { restores: ['qwen3VLEncoderModel'] },
  qwen3_vl_vision_tower: { restores: ['qwen3VLVisionTowerModel'] },
  qwen_image_qwen_vl_encoder: { restores: ['qwenVLEncoderModel'] },
  wan_t5_encoder_model: { restores: ['wanT5EncoderModel'] },
  qwen3_source: { restores: ['componentSourceModel'], when: unlessMainModel('qwen3_source') },
  qwen_image_component_source: {
    restores: ['componentSourceModel'],
    when: unlessMainModel('qwen_image_component_source'),
  },
  wan_component_source: { restores: ['componentSourceModel'], when: unlessMainModel('wan_component_source') },
  wan_transformer_low_noise: { restores: ['wanLowNoiseModel'] },
  ideogram4_unconditional_model: { restores: ['ideogram4UnconditionalModel'] },
  pid_decoder: { restores: ['pidDecoderModel'] },
  gemma2_encoder: { restores: ['gemma2EncoderModel'] },
  anima_inpaint_adapter: {
    absent: (model, written) =>
      model.base === 'anima' && ANIMA_ADAPTER_MODES.includes(written.generation_mode)
        ? { animaInpaintAdapterEnabled: false }
        : null,
    restores: ['animaInpaintAdapterModel', 'animaInpaintAdapterEnabled'],
  },
  anima_inpaint_adapter_weight: { restores: ['animaInpaintAdapterWeight'] },

  // Concepts and references.
  loras: { absent: () => ({ loras: [] }), restores: ['loras'] },
  ref_images: { restores: ['referenceImages'] },
  krea2_style_strength: {
    ignored: 'Repeats the style reference strength, which `ref_images` restores with the reference itself.',
  },

  // SDXL refiner.
  refiner_model: {
    absent: (model) => (getGenerationUiPolicy(model).refinerVisible ? { refinerModel: null } : null),
    restores: ['refinerModel'],
  },
  refiner_steps: { restores: ['refinerSteps'] },
  refiner_cfg_scale: { restores: ['refinerCfgScale'] },
  refiner_scheduler: { restores: ['refinerScheduler'] },
  refiner_start: { restores: ['refinerStart'] },
  refiner_positive_aesthetic_score: { restores: ['refinerPositiveAestheticScore'] },
  refiner_negative_aesthetic_score: { restores: ['refinerNegativeAestheticScore'] },

  // Family tuning.
  dype_preset: {
    absent: (model) => (model.base === 'flux' ? { fluxDypePreset: 'off' } : null),
    restores: ['fluxDypePreset'],
  },
  dype_scale: { restores: ['fluxDypeScale'] },
  dype_exponent: { restores: ['fluxDypeExponent'] },
  z_image_shift: {
    absent: (model) => (model.base === 'z-image' ? { zImageShift: null } : null),
    restores: ['zImageShift'],
  },
  z_image_seed_variance_enabled: {
    absent: (model) => (model.base === 'z-image' ? { zImageSeedVarianceEnabled: false } : null),
    restores: ['zImageSeedVarianceEnabled'],
  },
  z_image_seed_variance_strength: { restores: ['zImageSeedVarianceStrength'] },
  z_image_seed_variance_randomize_percent: { restores: ['zImageSeedVarianceRandomizePercent'] },
  qwen_image_shift: {
    absent: (model) => (model.base === 'qwen-image' ? { qwenImageShift: null } : null),
    restores: ['qwenImageShift'],
  },
  qwen_image_quantization: {
    absent: (model) => (model.base === 'qwen-image' ? { qwenImageEncoderQuantization: 'none' } : null),
    restores: ['qwenImageEncoderQuantization'],
  },
  wan_guidance_scale_low_noise: {
    absent: (model) => (model.base === 'wan' ? { wanGuidanceScaleLowNoise: null } : null),
    restores: ['wanGuidanceScaleLowNoise'],
  },
  ideogram4_sampler_preset: { restores: ['ideogram4SamplerPreset'] },
  ideogram4_steps: {
    absent: (model) => (model.base === 'ideogram-4' ? { ideogram4Steps: null } : null),
    restores: ['ideogram4Steps'],
  },
  ideogram4_guidance_scale: {
    absent: (model) => (model.base === 'ideogram-4' ? { ideogram4GuidanceScale: null } : null),
    restores: ['ideogram4GuidanceScale'],
  },
  ideogram4_mu: {
    absent: (model) => (model.base === 'ideogram-4' ? { ideogram4Mu: null } : null),
    restores: ['ideogram4Mu'],
  },
  ideogram4_color_palette: { restores: ['ideogram4ColorPalette'] },
  ideogram4_caption: { ignored: RUNTIME_VALUE },
  krea2_rebalance_enabled: { restores: ['krea2RebalanceEnabled'] },
  krea2_rebalance_multiplier: { restores: ['krea2RebalanceMultiplier'] },
  krea2_rebalance_weights: { restores: ['krea2RebalanceWeights'] },
  krea2_seed_variance_enabled: { restores: ['krea2SeedVarianceEnabled'] },
  krea2_seed_variance_strength: { restores: ['krea2SeedVarianceStrength'] },
  krea2_seed_variance_randomize_percent: { restores: ['krea2SeedVarianceRandomizePercent'] },
  krea2_reference_kv_cache: { restores: ['krea2ReferenceKvCache'] },
  moss_image_uncond_skip_layers: { restores: ['mossImageUncondSkipLayers'] },
  moss_image_layout_prompt: { ignored: RUNTIME_VALUE },
  flux3_caption: { ignored: RUNTIME_VALUE },
  pid_mode: {
    absent: (model) => (getGenerationUiPolicy(model).pidVisible ? { pidMode: 'off' } : null),
    restores: ['pidMode'],
  },
  pid_steps: { restores: ['pidSteps'] },

  // Hosted providers. A named size restores only with the size it names: a Canvas frame keeps its own size.
  image_size: {
    restores: ['externalImageSize', 'width', 'height', 'aspectRatioId', 'aspectRatioValue'],
    when: (model, written) =>
      getExternalResolutionPresets(model).some(
        (preset) =>
          preset.image_size === written.image_size && preset.width === written.width && preset.height === written.height
      ),
  },
  openai_quality: { restores: ['openaiQuality'] },
  openai_background: { restores: ['openaiBackground'] },
  openai_input_fidelity: { restores: ['openaiInputFidelity'] },
  gemini_temperature: { restores: ['geminiTemperature'] },
  gemini_thinking_level: { restores: ['geminiThinkingLevel'] },
  seedream_watermark: { restores: ['seedreamWatermark'] },
  seedream_optimize_prompt: { restores: ['seedreamOptimizePrompt'] },

  // Canvas.
  strength: { consumed: 'Canvas recall restores the denoising strength; it also lets a Canvas record offer recall.' },
  control_layers: {
    consumed: 'Canvas recall restores the control layers; they also let a Canvas record offer recall.',
  },
  controlnets: {
    consumed: 'The backend-declared view of `control_layers`; Canvas recall reads it from records without them.',
  },
  t2iAdapters: {
    consumed: 'The backend-declared view of `control_layers`; Canvas recall reads it from records without them.',
  },
  canvas_v3_metadata: { consumed: 'Canvas recall restores the Canvas document from it.' },
  regions: {
    ignored:
      'The resolved regional prompts and masks, recorded as provenance; Canvas recall restores the regional ' +
      'guidance layers from `canvas_v3_metadata` instead.',
  },

  // Upscale.
  creativity: { ignored: UPSCALE_SETTING },
  structure: { ignored: UPSCALE_SETTING },
  tile_overlap: { ignored: UPSCALE_SETTING },
  tile_size: { ignored: UPSCALE_SETTING },
  upscale_initial_image: { ignored: UPSCALE_SETTING },
  upscale_model: { ignored: UPSCALE_SETTING },
  upscale_scale: { ignored: UPSCALE_SETTING },

  // Declared by the backend, written by no image graph.
  ipAdapters: {
    declaredOnly: true,
    ignored: 'Written by the legacy web UI; recall reads it from those records as reference images.',
  },
  init_image: { declaredOnly: true, ignored: 'Legacy image-to-image source; the Canvas records its own.' },
  hrf_enabled: { declaredOnly: true, ignored: 'Legacy high-resolution fix, which Generate does not offer.' },
  hrf_method: { declaredOnly: true, ignored: 'Legacy high-resolution fix, which Generate does not offer.' },
  hrf_strength: { declaredOnly: true, ignored: 'Legacy high-resolution fix, which Generate does not offer.' },
  positive_style_prompt: {
    declaredOnly: true,
    ignored: 'Legacy SDXL style prompt; the SDXL graph sends the prompt to both encoders.',
  },
  negative_style_prompt: {
    declaredOnly: true,
    ignored: 'Legacy SDXL style prompt; the SDXL graph sends the prompt to both encoders.',
  },
  num_frames: { declaredOnly: true, ignored: VIDEO_PROFILE },
  fps: { declaredOnly: true, ignored: VIDEO_PROFILE },
  first_frame_image: { declaredOnly: true, ignored: VIDEO_PROFILE },
  last_frame_image: { declaredOnly: true, ignored: VIDEO_PROFILE },
  source_video: { declaredOnly: true, ignored: VIDEO_PROFILE },
  source_video_start_frame: { declaredOnly: true, ignored: VIDEO_PROFILE },
  source_video_end_frame: { declaredOnly: true, ignored: VIDEO_PROFILE },
  ltx2_context_frames: { declaredOnly: true, ignored: VIDEO_PROFILE },
  minimax_h3_transformer_model: { declaredOnly: true, ignored: VIDEO_PROFILE },
  minimax_h3_text_encoder_model: { declaredOnly: true, ignored: VIDEO_PROFILE },
  minimax_h3_component_source: { declaredOnly: true, ignored: VIDEO_PROFILE },
  minimax_h3_hybrid_base_model: { declaredOnly: true, ignored: VIDEO_PROFILE },
  minimax_h3_hybrid_start_block: { declaredOnly: true, ignored: VIDEO_PROFILE },
  minimax_h3_references: { declaredOnly: true, ignored: VIDEO_PROFILE },
} satisfies Record<DeclaredCoreMetadataKey, Entry> & Record<string, Entry>;

const CONTRACT: Readonly<Record<string, Entry>> = IMAGE_RECALL_CONTRACT;

type Contract = typeof IMAGE_RECALL_CONTRACT;

/** The keys the table calls declared only, which the backend schema must still declare. */
type DeclaredOnlyKey = {
  [K in keyof Contract]: Contract[K] extends { declaredOnly: true } ? K : never;
}[keyof Contract];

const entryOf = (key: string): Entry | undefined => (Object.hasOwn(CONTRACT, key) ? CONTRACT[key] : undefined);

/** Every label recall can report, so G5 cannot miss a new one: an unlisted member fails `tsc`. */
const RECALLED_FIELDS = {
  animaInpaintAdapterWeight: true,
  cfg: true,
  clipSkip: true,
  components: true,
  externalProvider: true,
  hiDiffusion: true,
  krea2Rebalance: true,
  krea2SeedVariance: true,
  loras: true,
  model: true,
  modelTuning: true,
  mossImageSkipLayers: true,
  prompts: true,
  referenceImages: true,
  refiner: true,
  scheduler: true,
  seamless: true,
  seed: true,
  size: true,
  steps: true,
  vae: true,
} satisfies Record<ImageRecallResult['fields'][number], true>;

const DEFAULT_KINDS: readonly ImageRecallKind[] = ['all', 'remix'];

/** The settings choosing a model resets to its defaults before the record lands, which a vacuous round trip hides. */
const MODEL_DEFAULT_SETTINGS: ReadonlySet<Setting> = new Set([
  'aspectRatioId',
  'aspectRatioValue',
  'cfgRescaleMultiplier',
  'cfgScale',
  'height',
  'scheduler',
  'steps',
  'width',
]);

/** Values the policy holds at the model default, so every record carries the default. */
const isPolicyPinned = (model: GenerateModelConfig, setting: Setting): boolean =>
  // FLUX.3 Klein samples without CFG; validation refuses any other scale.
  (setting === 'cfgScale' && model.base === 'flux3' && model.variant === 'flux3_klein') ||
  // A family without a scheduler choice records its fixed one.
  (setting === 'scheduler' && !getGenerationUiPolicy(model).schedulerVisible);

const BASELINE_MODEL: MainModelConfig = { base: 'sdxl', key: 'baseline-sdxl', name: 'Baseline SDXL', type: 'main' };
const SD1_BASELINE_MODEL: MainModelConfig = {
  base: 'sd-1',
  key: 'baseline-sd-1',
  name: 'Baseline SD 1.5',
  type: 'main',
};

/** The panel recall lands on: SDXL defaults with a concept, a prompt template and a random seed. */
const baselineValues = (): GenerateWidgetValues => ({
  ...getDefaultGenerateSettings(BASELINE_MODEL),
  loras: [
    { isEnabled: true, model: { base: 'sdxl', key: 'baseline-lora', name: 'Baseline LoRA', type: 'lora' }, weight: 1 },
  ],
  model: BASELINE_MODEL,
  positivePrompt: 'the baseline prompt',
  promptTemplate: { id: 'template', name: 'Template', negativePrompt: 'lowres', positivePrompt: '{prompt}, detailed' },
  promptTemplateViewMode: true,
  seed: 123,
  seedMode: 'random',
});

/** CLIP skip is judged against the panel's own model, which SDXL lacks. */
const sd1BaselineValues = (): GenerateWidgetValues => ({
  ...getDefaultGenerateSettings(SD1_BASELINE_MODEL),
  clipSkip: 5,
  model: SD1_BASELINE_MODEL,
  seed: 123,
});

/** The baseline off the default of every setting a generation record resets when it lacks the key (G8). */
const offDefaultBaselineValues = (): GenerateWidgetValues => ({
  ...baselineValues(),
  fluxDypePreset: 'manual',
  ideogram4GuidanceScale: 5,
  ideogram4Mu: 1,
  ideogram4Steps: 30,
  pidMode: 'fit',
  qwenImageEncoderQuantization: 'nf4',
  qwenImageShift: 2,
  refinerModel: { base: 'sdxl-refiner', key: 'baseline-refiner', name: 'Baseline refiner', type: 'main' },
  wanGuidanceScaleLowNoise: 3.5,
  zImageSeedVarianceEnabled: true,
  zImageShift: 2,
});

const imageOf = ({ height, width }: { height: number; width: number }): GalleryImage => ({
  boardId: 'none',
  height,
  imageCategory: 'general',
  imageName: 'recalled.png',
  imageUrl: '/recalled.png',
  queuedAt: '2026-11-09T00:00:00.000Z',
  sourceQueueItemId: 'recall-contract',
  starred: false,
  thumbnailUrl: '/recalled-thumbnail.png',
  width,
});

const PROJECT_SETTINGS = { useCpuNoise: true, useNsfwChecker: false, useWatermark: false };

type RecallInput = Parameters<typeof buildImageRecallSettings>[0];

interface RecallCase {
  label: string;
  /** The main model the record names, which recall settles on. */
  model: GenerateModelConfig;
  record: Written;
  catalog: Pick<RecallInput, 'models' | 'supportedModels' | 'vaeModels'>;
  image: GalleryImage;
  /** The Generate settings the record was compiled from; null for a record of another panel's settings. */
  tuned: GenerateWidgetValues | null;
  /** The panel recall lands on. */
  baseline: GenerateWidgetValues;
  isCanvasFrame?: true;
}

interface SettingsFixture {
  label: string;
  model: GenerateModelConfig;
  settings: GenerateSettings;
}

/** Compiles the fixture, naming it when validation refuses it. */
const compiled = <T>(label: string, compile: () => T): T => {
  try {
    return compile();
  } catch (error) {
    throw new Error(`${label} does not compile: ${String(error)}`, { cause: error });
  }
};

const generateCase = ({ label, model, settings }: SettingsFixture, baseline: GenerateWidgetValues): RecallCase => ({
  baseline,
  catalog: recallCatalogFor(model, settings),
  image: imageOf(settings),
  label,
  model,
  record: recordOf(
    compiled(label, () => compileGenerateGraph(settings, model, 'gallery', PROJECT_SETTINGS)),
    settings
  ),
  tuned: { ...settings, model },
});

const canvasCase = (
  { label, model, settings }: SettingsFixture,
  mode: CanvasCompileMode,
  baseline: GenerateWidgetValues,
  extra: Omit<NonNullable<Parameters<typeof compileCanvas>[2]>, 'bbox' | 'settings'> & {
    frame?: { height: number; width: number };
    isCanvasFrame?: true;
  } = {}
): RecallCase => {
  const { frame = settings, isCanvasFrame, ...overrides } = extra;
  const caseLabel = `${label} / canvas ${mode}`;
  // PiD decodes only the modes that keep the whole frame.
  const canvasSettings = mode === 'txt2img' || mode === 'img2img' ? settings : { ...settings, pidMode: 'off' as const };

  return {
    baseline,
    catalog: recallCatalogFor(model, canvasSettings),
    image: imageOf(frame),
    ...(isCanvasFrame ? { isCanvasFrame } : {}),
    label: caseLabel,
    model,
    record: recordOf(
      compiled(caseLabel, () =>
        compileCanvas(model, mode, {
          ...overrides,
          bbox: { height: frame.height, width: frame.width, x: 0, y: 0 },
          settings: canvasSettings,
        })
      ),
      canvasSettings
    ),
    // The record names the frame's size.
    tuned: {
      ...canvasSettings,
      aspectRatioId: deriveAspectRatioId(frame.width, frame.height),
      aspectRatioValue: frame.width / frame.height,
      height: frame.height,
      model,
      width: frame.width,
    },
  };
};

const upscaleCase = (testCase: (typeof UPSCALE_CASES)[number], baseline: GenerateWidgetValues): RecallCase => {
  const label = `upscale / ${testCase.label}`;
  const { installed, values } = valuesForCase(testCase);
  const [model] = installed.filter(isSupportedGenerateModel);

  return {
    baseline,
    catalog: {
      models: installed,
      supportedModels: [model!],
      vaeModels: installed.flatMap((entry) => (isVaeModelConfig(entry) ? [entry] : [])),
    },
    image: imageOf({ height: 400, width: 808 }),
    label,
    model: model!,
    record: recordOf(
      compiled(label, () => compileUpscaleGraph(values, 'gallery', { useCpuNoise: true })),
      values
    ),
    tuned: null,
  };
};

/** Model only copies the source's record, as the backend saved it, and adds what the upscale did. */
const modelOnlyUpscaleCase = (source: RecallCase): RecallCase => {
  const { backendGraph } = compileModelOnlyUpscaleGraph({
    destination: 'gallery',
    image: { height: source.image.height, image_name: source.image.imageName, width: source.image.width },
    scale: 2,
    sourceMetadata: { ...source.record, app_version: '7.0.0', metadata_version: '1.2.0' },
    upscaleModel: { ...INSTALLED[0], scale: 4 } as SpandrelModelConfig,
  });

  return {
    ...source,
    image: imageOf({ height: source.image.height * 2, width: source.image.width * 2 }),
    label: `upscale / model-only from ${source.label}`,
    record: JSON.parse(JSON.stringify(backendGraph.nodes.upscale_output!.metadata)) as Written,
  };
};

const ANIMA_INPAINT_ADAPTER: ComponentModelConfig = {
  base: 'anima',
  cond_in_channels: 4,
  key: 'anima-inpaint-adapter',
  name: 'Anima LLLite Inpainting',
  type: 'controlnet',
};

const CANVAS_REGION = {
  autoNegative: false,
  id: 'region',
  maskBounds: { height: 256, width: 192, x: 64, y: 128 },
  maskImageName: 'region-mask.png',
  negativePrompt: null,
  positivePrompt: 'a cat',
  referenceImages: [],
};

const buildCases = (): RecallCase[] => {
  const baseline = baselineValues();
  const tunedShapes = generateGraphCases.map(({ base, shape }) => recallTunedSettingsFor(base, shape));
  const generate = [
    ...tunedShapes,
    sdxlRefinerRecallCase(),
    fluxReduxRecallCase(),
    krea2EditRecallCase(),
    ...externalRecallCases(),
  ].map((fixture) => generateCase(fixture, baseline));
  // One shape per base: the Canvas adds the same keys to every shape's record.
  const firstShapes = tunedShapes.filter(
    (fixture, index) => tunedShapes.findIndex((other) => other.model.base === fixture.model.base) === index
  );
  const fixtureFor = (base: string) => firstShapes.find((fixture) => fixture.model.base === base)!;
  const canvas = firstShapes.flatMap((fixture) =>
    canvasModesFor(fixture.model, fixture.settings).map((mode) => canvasCase(fixture, mode, baseline))
  );
  const anima = fixtureFor('anima');
  const sdxl = fixtureFor('sdxl');
  const [, gemini] = externalRecallCases();

  return [
    ...generate,
    ...canvas,
    // The adapter switch is on by default, so only a panel that has it off shows a record turning it on.
    canvasCase(
      {
        ...anima,
        label: `${anima.label} with inpaint adapter`,
        settings: {
          ...anima.settings,
          animaInpaintAdapterEnabled: true,
          animaInpaintAdapterModel: ANIMA_INPAINT_ADAPTER,
          animaInpaintAdapterWeight: 0.7,
        },
      },
      'inpaint',
      { ...baseline, animaInpaintAdapterEnabled: false }
    ),
    canvasCase({ ...sdxl, label: `${sdxl.label} with layers` }, 'img2img', baseline, {
      canvasMetadata: { version: 1 },
      controlLayers: [
        {
          beginEndStepPct: [0.1, 0.85],
          controlMode: 'more_control',
          id: 'control',
          imageName: 'control.png',
          kind: 'controlnet',
          model: { base: 'sdxl', key: 'sdxl-controlnet', name: 'SDXL ControlNet', type: 'controlnet' },
          weight: 0.65,
        },
        {
          beginEndStepPct: [0, 1],
          controlMode: null,
          id: 't2i',
          imageName: 't2i.png',
          kind: 't2i_adapter',
          model: { base: 'sdxl', key: 'sdxl-t2i-adapter', name: 'SDXL T2I-Adapter', type: 't2i_adapter' },
          weight: 0.8,
        },
      ],
      regionalGuidance: [CANVAS_REGION],
    }),
    // A layout family places regions in its caption, which the record takes from the caption node.
    canvasCase({ ...fixtureFor('flux3'), label: `${fixtureFor('flux3').label} with a region` }, 'txt2img', baseline, {
      regionalGuidance: [CANVAS_REGION],
    }),
    // The provider was asked for its named size nearest the frame; a Canvas route keeps the frame.
    canvasCase(gemini!, 'txt2img', baseline, { frame: { height: 800, width: 1000 }, isCanvasFrame: true }),
    ...UPSCALE_CASES.map((testCase) => upscaleCase(testCase, baseline)),
    modelOnlyUpscaleCase(generate.find(({ label }) => label === 'sdxl / diffusers')!),
  ];
};

let cases: RecallCase[] | undefined;

/** Built inside a test, where the architecture table is seeded. */
const getCases = (): RecallCase[] => (cases ??= buildCases());

const recall = (
  testCase: RecallCase,
  kind: ImageRecallKind,
  {
    currentValues = testCase.baseline,
    metadata = testCase.record,
  }: { currentValues?: GenerateWidgetValues; metadata?: Written } = {}
) =>
  buildImageRecallSettings({
    ...testCase.catalog,
    currentValues,
    image: testCase.image,
    isCanvasFrame: testCase.isCanvasFrame,
    kind,
    metadata,
  });

const equal = areJsonValuesStructurallyEqual;
const show = (value: unknown): string => JSON.stringify(value) ?? String(value);

/** The written keys whose entry restores a setting, with the entry. */
const restoringKeys = ({ record }: RecallCase): [string, Restores][] =>
  Object.keys(record).flatMap((key) => {
    const entry = entryOf(key);

    return entry && 'restores' in entry ? [[key, entry]] : [];
  });

/** The settings the record's restoring entries apply under `kind`. */
const settingsUnder = (testCase: RecallCase, kind: ImageRecallKind): Setting[] => [
  ...new Set(
    restoringKeys(testCase).flatMap(([, entry]) =>
      (entry.kinds ?? DEFAULT_KINDS).includes(kind) ? entry.restores : []
    )
  ),
];

const pick = (values: GenerateWidgetValues, settings: readonly Setting[]): Partial<GenerateWidgetValues> =>
  Object.fromEntries(settings.map((setting) => [setting, values[setting]]));

const changedSettings = (left: GenerateWidgetValues, right: GenerateWidgetValues): string[] =>
  (Object.keys(left) as Setting[]).filter((setting) => !equal(left[setting], right[setting]));

seedArchitectureCapabilities();

describe('image recall contract', () => {
  it('G1: has an entry for every key an image graph writes', () => {
    const failures = getCases().flatMap(({ label, model, record }) =>
      Object.entries(record).flatMap(([key, value]) => {
        const entry = entryOf(key);

        if (!entry) {
          return [`G1: \`${key}\` written by \`${label}\` has no entry in IMAGE_RECALL_CONTRACT`];
        }

        return isRuntimeValue(value) && 'restores' in entry && entry.when?.(model, record) !== false
          ? [`G1: \`${key}\` in \`${label}\` is computed while the graph runs (${value}); no round trip can show it`]
          : [];
      })
    );

    expect(failures).toEqual([]);
  });

  it('G2: restores every value it claims to, from a panel that held another', () => {
    const failures: string[] = [];

    for (const testCase of getCases()) {
      const { baseline, label, model, record, tuned } = testCase;

      if (!tuned) {
        continue;
      }

      const values = recall(testCase, 'all')?.values;
      const modelDefaults: GenerateWidgetValues = { ...getDefaultGenerateSettings(model), model };

      for (const [key, entry] of restoringKeys(testCase)) {
        if (entry.when?.(model, record) === false) {
          continue;
        }

        const expected = { ...pick(tuned, entry.restores), ...entry.expected?.(record, tuned) };

        for (const setting of entry.restores) {
          if (!equal(values?.[setting], expected[setting])) {
            failures.push(
              `G2: \`${key}\` in \`${label}\` restores ${setting} as ${show(values?.[setting])}, expected ${show(expected[setting])}`
            );
          }

          if (
            equal(baseline[setting], expected[setting]) ||
            (MODEL_DEFAULT_SETTINGS.has(setting) &&
              equal(modelDefaults[setting], expected[setting]) &&
              !isPolicyPinned(model, setting))
          ) {
            failures.push(
              `G2 vacuous: \`${key}\` in \`${label}\` expects ${setting} ${show(expected[setting])}, which the panel ` +
                'or the model default already holds'
            );
          }
        }
      }
    }

    expect(failures).toEqual([]);
  });

  it('G3: leaves the settings as they are without each key it does not restore', () => {
    const failures: string[] = [];

    for (const testCase of getCases()) {
      const { label, model, record } = testCase;
      const full = recall(testCase, 'all')?.values ?? null;

      for (const key of Object.keys(record)) {
        const entry = entryOf(key);
        const inert = entry && ('ignored' in entry || ('restores' in entry && entry.when?.(model, record) === false));

        if (!inert) {
          continue;
        }

        const { [key]: _ablated, ...without } = record;
        const ablated = recall(testCase, 'all', { metadata: without })?.values ?? null;

        if (!equal(full, ablated)) {
          failures.push(
            `G3: \`${key}\` in \`${label}\` changes ${full && ablated ? changedSettings(full, ablated).join(', ') : 'whether anything is recalled'}, so recall reads it`
          );
        }
      }
    }

    expect(failures).toEqual([]);
  });

  it('G4: applies each recall kind to exactly its own settings', () => {
    const failures: string[] = [];
    const sd1Baseline = sd1BaselineValues();

    for (const testCase of getCases()) {
      const { baseline, image, label, record } = testCase;
      const all = recall(testCase, 'all')?.values;

      if (!all) {
        failures.push(`G4: \`${label}\` recalls nothing`);
        continue;
      }

      // Remix is Recall All without the settings only Recall All applies: the seed.
      const allOnly = settingsUnder(testCase, 'all').filter(
        (setting) => !settingsUnder(testCase, 'remix').includes(setting)
      );
      const expectations: [ImageRecallKind, GenerateWidgetValues | null, GenerateWidgetValues][] = [
        ['remix', recall(testCase, 'remix')?.values ?? null, { ...all, ...pick(baseline, allOnly) }],
        [
          'prompts',
          recall(testCase, 'prompts')?.values ?? null,
          { ...baseline, ...pick(all, settingsUnder(testCase, 'prompts')) },
        ],
        [
          'seed',
          recall(testCase, 'seed')?.values ?? null,
          { ...baseline, ...pick(all, settingsUnder(testCase, 'seed')) },
        ],
        [
          'dimensions',
          recall(testCase, 'dimensions')?.values ?? null,
          {
            ...baseline,
            aspectRatioId: deriveAspectRatioId(image.width, image.height),
            aspectRatioValue: image.width / image.height,
            height: image.height,
            width: image.width,
          },
        ],
      ];

      for (const [kind, actual, expected] of expectations) {
        if (!actual || !equal(actual, expected)) {
          failures.push(
            `G4: ${kind} recall of \`${label}\` ${actual ? `differs in ${changedSettings(actual, expected).join(', ')}` : 'restores nothing'}`
          );
        }
      }

      // SD 1.5 caps CLIP skip at 12.
      const clipSkip = recall(testCase, 'clipSkip', { currentValues: sd1Baseline })?.values ?? null;
      const expectedClipSkip =
        typeof record.clip_skip === 'number' ? { ...sd1Baseline, clipSkip: Math.min(12, record.clip_skip) } : null;

      if (!equal(clipSkip, expectedClipSkip)) {
        failures.push(
          `G4: clipSkip on \`${label}\` gives ${show(clipSkip?.clipSkip)}, expected ${show(expectedClipSkip?.clipSkip)}`
        );
      }
    }

    expect(failures).toEqual([]);
  });

  it('G5: reports every field label from some record', () => {
    const reported = new Set<string>(getCases().flatMap((testCase) => recall(testCase, 'all')?.fields ?? []));

    expect(
      Object.keys(RECALLED_FIELDS).filter((field) => !reported.has(field)),
      'G5: no case reports these fields'
    ).toEqual([]);
  });

  it('G6: offers every recall a record supports', () => {
    const sd1Baseline = sd1BaselineValues();
    const failures = getCases().flatMap((testCase) => {
      const { baseline, catalog, image, label, record } = testCase;
      const input = { ...catalog, image, metadata: record };
      const capabilities = getImageRecallCapabilities({ ...input, currentValues: baseline });
      const expected = {
        all: true,
        clipSkip: false,
        dimensions: true,
        prompts: true,
        remix: true,
        seed: true,
        workflow: false,
      };
      const sd1ClipSkip = getImageRecallCapabilities({ ...input, currentValues: sd1Baseline }).clipSkip;

      return [
        ...(equal(capabilities, expected) ? [] : [`G6: \`${label}\` offers ${show(capabilities)}`]),
        ...(sd1ClipSkip === 'clip_skip' in record
          ? []
          : [`G6: \`${label}\` offers CLIP skip ${String(sd1ClipSkip)} on SD 1.5`]),
      ];
    });

    expect(failures).toEqual([]);
  });

  it('G7: keeps no entry no record exercises', () => {
    // Checked by `tsc`: a key the backend stops declaring fails here until its entry goes.
    expectTypeOf<Exclude<DeclaredOnlyKey, DeclaredCoreMetadataKey>>().toEqualTypeOf<never>();

    const allCases = getCases();
    const written = new Set(allCases.flatMap(({ record }) => Object.keys(record)));
    const roundTripped = new Set(
      allCases.flatMap((testCase) =>
        testCase.tuned
          ? restoringKeys(testCase)
              .filter(([, entry]) => entry.when?.(testCase.model, testCase.record) !== false)
              .map(([key]) => key)
          : []
      )
    );
    const failures = Object.entries(CONTRACT).flatMap(([key, entry]) => {
      if ('restores' in entry) {
        return roundTripped.has(key) ? [] : [`G7: no record round-trips \`${key}\``];
      }

      if ('ignored' in entry && entry.declaredOnly) {
        return written.has(key) ? [`G7: \`${key}\` is written by an image graph, so it is not declared only`] : [];
      }

      return written.has(key) ? [] : [`G7: no image graph writes \`${key}\``];
    });

    expect(failures).toEqual([]);
  });

  it('G8: restores the default a generation record implies for each key it lacks', () => {
    const offDefaults = offDefaultBaselineValues();
    const failures: string[] = [];

    for (const testCase of getCases()) {
      const { label, model, record } = testCase;

      if (typeof record.generation_mode !== 'string') {
        continue;
      }

      const values = recall(testCase, 'all', { currentValues: offDefaults })?.values;

      for (const [key, entry] of Object.entries(CONTRACT)) {
        const defaults = !(key in record) && 'restores' in entry ? entry.absent?.(model, record) : null;

        for (const [setting, value] of Object.entries(defaults ?? {}) as [Setting, unknown][]) {
          if (!equal(values?.[setting], value)) {
            failures.push(`G8: \`${label}\` lacks \`${key}\` but keeps ${setting} ${show(values?.[setting])}`);
          }

          if (equal(offDefaults[setting], value)) {
            failures.push(
              `G8 vacuous: \`${label}\` lacks \`${key}\`, but the panel already holds ${setting} ${show(value)}`
            );
          }
        }
      }
    }

    expect(failures).toEqual([]);
  });
});
