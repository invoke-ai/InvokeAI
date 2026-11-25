import type {
  ExpandPromptSuggestion,
  GenerateLora,
  GenerationModelCatalogItem as ModelConfig,
  GenerationModelTaxonomyType as ModelTaxonomyType,
  ImageWithDims,
  LoraModelConfig,
  MainModelConfig,
} from '@features/generation/contracts';

import {
  getCompatibleDiffusersComponentSource,
  isDiffusersMainForBase,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isWanLoraTargetingMain,
} from '@features/generation/settings';
import { SEED_MAX } from '@platform/core/seed';

import type {
  Ltx2TargetResolution,
  MiniMaxH3TargetResolution,
  VideoAspectRatioId,
  VideoGenerationMode,
  VideoSettings,
  VideoTargetResolution,
} from './types';

import {
  getVideoAspectRatioParts,
  LTX2_CANVAS_MULTIPLE,
  LTX2_DEFAULT_NEGATIVE_PROMPT,
  LTX2_FPS_DEFAULT,
  LTX2_FPS_MAX,
  LTX2_FPS_MIN,
  LTX2_NUM_FRAMES_DEFAULT,
  LTX2_NUM_FRAMES_MAX,
  LTX2_NUM_FRAMES_MIN,
  LTX2_EXTEND_CONTEXT_FRAMES,
  ltx2MaxExtendContextFrames,
  ltx2NewFramesForExtend,
  LTX2_NUM_FRAMES_STEP,
  ltx2FramesForClip,
  MINIMAX_H3_FPS,
  MINIMAX_H3_NUM_FRAMES_CHOICES,
  MINIMAX_H3_NUM_FRAMES_DEFAULT,
  resolveLtx2Canvas,
  resolveMiniMaxH3Canvas,
  scaleAndSnapWanDimensions,
  snapLtx2FramesDown,
  snapNumFramesToChoices,
  snapNumFramesToGrid,
  WAN_A14B_PIXEL_MULTIPLE,
  WAN_FPS_DEFAULT,
  WAN_FPS_MAX,
  WAN_FPS_MIN,
  WAN_NUM_FRAMES_DEFAULT,
  WAN_NUM_FRAMES_MAX,
  WAN_NUM_FRAMES_MIN,
  WAN_NUM_FRAMES_STEP,
  WAN_TI2V_PIXEL_MULTIPLE,
  ltx2AutoDurationBounds,
  type VideoDimensions,
  type VideoFramesChoices,
  type VideoFramesGrid,
} from './dimensions';
import {
  applyReferenceExtendSourceVideo,
  applyReferenceExtendNumFrames,
  isConditioningClipExcludingFrames,
  isVideoSourceClipTrimmed,
  REFERENCE_EXTEND_ANCHOR_CONDITIONING,
  MIN_VIDEO_TRIM_FRAMES,
  MINIMAX_H3_HYBRID_BLOCK_RANGE,
  resolveVideoMode,
  VIDEO_ASPECT_RATIO_IDS,
  VIDEO_DYNAMIC_PROMPTS_DEFAULTS,
} from './settings';

export type SupportedVideoBase = 'wan' | 'minimax-h3' | 'ltx-2';

/** Catalog records stay untouched; only task-dependent consumers use this derived model. */
export const getEffectiveVideoModel = <T extends MainModelConfig | undefined | null>(
  model: T,
  settings: Pick<VideoSettings, 'h3TaskOverride'>
): T => {
  if (model?.base !== 'minimax-h3' || model.format !== 'checkpoint') {
    return model;
  }
  const variant = settings.h3TaskOverride ?? (model.variant === 'ref2va' ? 'ref2va' : 'fl2va');
  return (variant === model.variant ? model : { ...model, variant }) as T;
};

export type VideoNegativePromptUsage = 'always' | 'cfg-gated' | 'never';

export interface VideoTargetResolutionOption {
  id: VideoTargetResolution;
  label: string;
  /** Absent means one pass. Two stages render at half this canvas, upscale the latent x2, then refine. */
  stages?: 2;
}

export interface VideoFramesGridPolicy extends VideoFramesGrid {
  kind: 'grid';
}

export interface VideoFramesChoicesPolicy extends VideoFramesChoices {
  kind: 'choices';
}

export type VideoFramesPolicy = VideoFramesGridPolicy | VideoFramesChoicesPolicy;

export interface VideoFpsPolicy {
  editable: boolean;
  defaultValue: number;
  min: number;
  max: number;
}

/** `steps` describes the reference release; getAcceleratorSteps resolves release-specific schedules. */
export interface VideoAcceleratorConfig {
  label: 'Lightning' | 'Turbo' | 'Distilled';
  steps: number;
  /** For releases whose names carry no "N-step" token; first match wins. */
  stepOverrides?: readonly { pattern: RegExp; steps: number }[];
  cfgScale: number;
  cfgScaleLowNoise: number | null;
  /** Guidance scales beyond CFG the distilled model needs; absent leaves those settings alone. */
  guidance?: { audioCfgScale: number; modalityScale: number; stgScale: number };
}

interface VideoVariantConfig {
  modes: readonly VideoGenerationMode[];
  pixelMultiple: number;
  targetResolutions: readonly VideoTargetResolutionOption[];
  /** Two-stage refine-pass budget; absent inherits the base pass's, which on guided Dev takes hours. */
  refineSteps?: number;
  defaults: {
    targetResolution: VideoTargetResolution;
    steps: number;
    cfgScale: number;
    cfgScaleLowNoise: number | null;
    audioCfgScale: number | null;
    stgScale: number | null;
    modalityScale: number | null;
  };
  minSteps: number;
  stepsEditable: boolean;
  frames: VideoFramesPolicy;
  fps: VideoFpsPolicy;
  cfg: { visible: boolean; lowNoiseVisible: boolean };
  guidance: { audioVisible: boolean; stgVisible: boolean; modalityVisible: boolean };
  defaultNegativePrompt?: string;
  negativePrompt: { visible: boolean; usage: VideoNegativePromptUsage };
  promptEnhancer?: VideoPromptEnhancer;
  accelerator: VideoAcceleratorConfig | null;
  audioOutput: boolean;
  /** Ref2VA reference caps; present only on variants whose modes include 'reference'. */
  references?: { maxVideos: number; maxImages: number; extend?: boolean };
}

export interface VideoPromptEnhancer {
  /** Starter source of the enhancer model; an installed text LLM with this source is preselected. */
  modelSource: string;
  /** The starter model's name (backend `starter_models/ltx_2.py`). */
  modelName: string;
  /** Seeded system prompts (backend migration `2026_09_26_add_ltx2_system_prompts`). */
  textToVideoSystemPromptId: string;
  /** Describes the video from its first frame; used only when Expand Prompt sends that frame along. */
  imageToVideoSystemPromptId: string;
}

const NO_EXTRA_GUIDANCE = { audioVisible: false, modalityVisible: false, stgVisible: false } as const;
const NO_EXTRA_GUIDANCE_DEFAULTS = { audioCfgScale: null, modalityScale: null, stgScale: null } as const;

export const WAN_LIGHTNING_ACCELERATOR: VideoAcceleratorConfig = {
  cfgScale: 1,
  cfgScaleLowNoise: 1,
  label: 'Lightning',
  steps: 4,
};

// LightX2V H3 uses eight steps; starter display names omit that count but retain the organization token.
const LIGHTX2V_PATTERN = /(?:^|[^a-z0-9])lightx2v(?:[^a-z0-9]|$)/i;

export const MINIMAX_H3_TURBO_ACCELERATOR: VideoAcceleratorConfig = {
  cfgScale: 1,
  cfgScaleLowNoise: null,
  label: 'Turbo',
  steps: 6,
  stepOverrides: [{ pattern: LIGHTX2V_PATTERN, steps: 8 }],
};

/** `steps: 8` is stated because the release name carries only the rank (450). */
export const LTX2_DISTILLED_ACCELERATOR: VideoAcceleratorConfig = {
  cfgScale: 1,
  cfgScaleLowNoise: null,
  guidance: { audioCfgScale: 1, modalityScale: 1, stgScale: 0 },
  label: 'Distilled',
  steps: 8,
};

/** Ref2VA Turbo uses a four-step reference count; LightX2V releases override it to eight. */
export const MINIMAX_H3_REF2V_TURBO_ACCELERATOR: VideoAcceleratorConfig = {
  cfgScale: 1,
  cfgScaleLowNoise: null,
  label: 'Turbo',
  steps: 4,
  stepOverrides: [{ pattern: LIGHTX2V_PATTERN, steps: 8 }],
};

const WAN_TARGET_RESOLUTION_OPTIONS: readonly VideoTargetResolutionOption[] = [
  { id: '480p', label: '480p (Wan native)' },
  { id: '720p', label: '720p (Wan native)' },
  { id: '1080p', label: '1080p (extrapolated)' },
];

const MINIMAX_H3_TARGET_RESOLUTION_OPTIONS: readonly VideoTargetResolutionOption[] = [
  { id: '768 highres', label: '768 highres (H3 native)' },
  { id: '768 lowres', label: '768 lowres (fast preview)' },
  { id: '480p', label: '480p draft (short edge 480)' },
  { id: '544p', label: '544p draft (short edge 544)' },
];

const WAN_FRAMES: VideoFramesGridPolicy = {
  defaultValue: WAN_NUM_FRAMES_DEFAULT,
  kind: 'grid',
  max: WAN_NUM_FRAMES_MAX,
  min: WAN_NUM_FRAMES_MIN,
  step: WAN_NUM_FRAMES_STEP,
};

const WAN_FPS: VideoFpsPolicy = { defaultValue: WAN_FPS_DEFAULT, editable: true, max: WAN_FPS_MAX, min: WAN_FPS_MIN };

// wan_video_denoise defaults: guidance_scale=5.0 (high), guidance_scale_low_noise=4.0.
const WAN_A14B_COMMON = {
  accelerator: WAN_LIGHTNING_ACCELERATOR,
  cfg: { lowNoiseVisible: true, visible: true },
  defaults: {
    ...NO_EXTRA_GUIDANCE_DEFAULTS,
    cfgScale: 5,
    cfgScaleLowNoise: 4,
    steps: 40,
    targetResolution: '720p' as const,
  },
  fps: WAN_FPS,
  frames: WAN_FRAMES,
  guidance: NO_EXTRA_GUIDANCE,
  minSteps: 1,
  stepsEditable: true,
  audioOutput: false,
  negativePrompt: { usage: 'cfg-gated' as const, visible: true },
  pixelMultiple: WAN_A14B_PIXEL_MULTIPLE,
  targetResolutions: WAN_TARGET_RESOLUTION_OPTIONS,
};

const WAN_VARIANTS: Record<string, VideoVariantConfig> = {
  t2v_a14b: { ...WAN_A14B_COMMON, modes: ['txt2vid'] },
  // I2V experts require a reference frame; extension uses the source's last frame through FLF2V conditioning.
  i2v_a14b: { ...WAN_A14B_COMMON, modes: ['first-frame', 'first-last', 'extend'] },
  ti2v_5b: {
    ...WAN_A14B_COMMON,
    accelerator: null,
    cfg: { lowNoiseVisible: false, visible: true },
    defaults: {
      ...NO_EXTRA_GUIDANCE_DEFAULTS,
      cfgScale: 5,
      cfgScaleLowNoise: null,
      steps: 40,
      targetResolution: '720p',
    },
    modes: ['txt2vid', 'first-frame', 'extend'],
    pixelMultiple: WAN_TI2V_PIXEL_MULTIPLE,
  },
};

// Unknown Wan variants get the most permissive capabilities: the backend probe is the authority.
const WAN_FALLBACK_VARIANT: VideoVariantConfig = {
  ...WAN_A14B_COMMON,
  modes: ['txt2vid', 'first-frame', 'first-last', 'extend'],
};

// Guidance-distilled: no CFG, no negative prompt, fixed 24 fps, audio included.
const MINIMAX_H3_FL2VA: VideoVariantConfig = {
  accelerator: MINIMAX_H3_TURBO_ACCELERATOR,
  audioOutput: true,
  cfg: { lowNoiseVisible: false, visible: false },
  defaults: {
    ...NO_EXTRA_GUIDANCE_DEFAULTS,
    cfgScale: 1,
    cfgScaleLowNoise: null,
    steps: 50,
    targetResolution: '768 highres',
  },
  fps: { defaultValue: MINIMAX_H3_FPS, editable: false, max: MINIMAX_H3_FPS, min: MINIMAX_H3_FPS },
  frames: { choices: MINIMAX_H3_NUM_FRAMES_CHOICES, defaultValue: MINIMAX_H3_NUM_FRAMES_DEFAULT, kind: 'choices' },
  guidance: NO_EXTRA_GUIDANCE,
  minSteps: 2,
  stepsEditable: true,
  modes: ['txt2vid', 'first-frame', 'last-frame', 'first-last', 'extend'],
  negativePrompt: { usage: 'never', visible: false },
  pixelMultiple: 32,
  targetResolutions: MINIMAX_H3_TARGET_RESOLUTION_OPTIONS,
};

const MINIMAX_H3_REF2VA: VideoVariantConfig = {
  ...MINIMAX_H3_FL2VA,
  accelerator: MINIMAX_H3_REF2V_TURBO_ACCELERATOR,
  modes: ['reference'],
  // Reference extension derives a linked tail reference from Initial Video; it does not frame-condition.
  references: { extend: true, maxImages: 9, maxVideos: 3 },
};

/** Below this the refine pass cannot enter its schedule anywhere near the level it asks for. */
export const LTX2_MIN_TWO_STAGE_STEPS = 2;

const LTX2_TARGET_RESOLUTION_OPTIONS: readonly VideoTargetResolutionOption[] = [
  { id: '512p', label: '512p (fastest)' },
  { id: '704p', label: '704p' },
  { id: '768p', label: '768p (sharpest)' },
  { id: '1024p', label: '1024p (2-stage)', stages: 2 },
  { id: '1536p', label: '1536p (2-stage, very slow)', stages: 2 },
];

const LTX2_FRAMES: VideoFramesGridPolicy = {
  defaultValue: LTX2_NUM_FRAMES_DEFAULT,
  kind: 'grid',
  max: LTX2_NUM_FRAMES_MAX,
  min: LTX2_NUM_FRAMES_MIN,
  step: LTX2_NUM_FRAMES_STEP,
};

const LTX2_COMMON = {
  accelerator: null,
  promptEnhancer: {
    imageToVideoSystemPromptId: '0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0010',
    modelName: 'LTX-2.5 Prompt Enhancer (Gemma-4 E2B)',
    modelSource: 'google/gemma-4-E2B-it',
    textToVideoSystemPromptId: '0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0009',
  },
  audioOutput: true,
  cfg: { lowNoiseVisible: false, visible: true },
  fps: { defaultValue: LTX2_FPS_DEFAULT, editable: true, max: LTX2_FPS_MAX, min: LTX2_FPS_MIN },
  frames: LTX2_FRAMES,
  minSteps: 1,
  modes: ['txt2vid', 'first-frame', 'last-frame', 'first-last', 'extend', 'audio-to-video', 'video-to-audio'] as const,
  pixelMultiple: LTX2_CANVAS_MULTIPLE,
  targetResolutions: LTX2_TARGET_RESOLUTION_OPTIONS,
};

/** Scales are ltx-pipelines `PipelineParams`; audio CFG is deliberately far above the video's. */
const LTX2_DEV: VideoVariantConfig = {
  ...LTX2_COMMON,
  // Only Dev: on the distilled checkpoint the LoRA would patch a model the distillation was not fitted to.
  accelerator: LTX2_DISTILLED_ACCELERATOR,
  // Mirrors LTX2_DEFAULT_NEGATIVE_PROMPT in invokeai/backend/ltx2/constants.py; the release guides against it.
  defaultNegativePrompt: LTX2_DEFAULT_NEGATIVE_PROMPT,
  defaults: {
    audioCfgScale: 7,
    cfgScale: 3,
    cfgScaleLowNoise: null,
    modalityScale: 3,
    steps: 30,
    stgScale: 1,
    targetResolution: '704p',
  },
  guidance: { audioVisible: true, modalityVisible: true, stgVisible: true },
  negativePrompt: { usage: 'cfg-gated', visible: true },
  // Dev pays four forwards per step, so the refine pass samples its own 8 steps rather than the base 30.
  refineSteps: 8,
  stepsEditable: true,
};

/** The denoise node ignores guidance scales on this checkpoint, so the panel hides them. */
const LTX2_DISTILLED: VideoVariantConfig = {
  ...LTX2_COMMON,
  cfg: { lowNoiseVisible: false, visible: false },
  defaults: {
    ...NO_EXTRA_GUIDANCE_DEFAULTS,
    cfgScale: 1,
    cfgScaleLowNoise: null,
    steps: 8,
    targetResolution: '704p',
  },
  guidance: NO_EXTRA_GUIDANCE,
  negativePrompt: { usage: 'never', visible: false },
  stepsEditable: false,
};

export const VIDEO_GENERATION: Record<
  SupportedVideoBase,
  { variants: Record<string, VideoVariantConfig>; fallback: VideoVariantConfig }
> = {
  // Dev is the fallback: a guided schedule is only slow on a distilled checkpoint; the reverse produces noise.
  'ltx-2': { fallback: LTX2_DEV, variants: { ltx2_dev: LTX2_DEV, ltx2_distilled: LTX2_DISTILLED } },
  'minimax-h3': { fallback: MINIMAX_H3_FL2VA, variants: { fl2va: MINIMAX_H3_FL2VA, ref2va: MINIMAX_H3_REF2VA } },
  wan: { fallback: WAN_FALLBACK_VARIANT, variants: WAN_VARIANTS },
};

export const SUPPORTED_VIDEO_BASES = Object.keys(VIDEO_GENERATION) as SupportedVideoBase[];

/** A single-file main owns the task identity; a Diffusers source supplies its remaining components. */
export const isSupportedVideoModel = <T extends { base: string; type: string; format?: string }>(
  model: T
): model is T & MainModelConfig =>
  model.type === 'main' &&
  (model.base === 'wan' ||
    (model.base === 'minimax-h3' && (model.format === 'diffusers' || model.format === 'checkpoint')) ||
    (model.base === 'ltx-2' &&
      (model.format === 'diffusers' || model.format === 'checkpoint' || model.format === 'gguf_quantized')));

/** Components-only and Ref2VA Diffusers folders stay valid stored state but serve only as component sources. */
export const isVideoModelSelectable = <T extends ModelConfig>(model: T): boolean =>
  isSupportedVideoModel(model) &&
  !isComponentsOnlyVideoMain(model) &&
  !(model.base === 'minimax-h3' && model.format === 'diffusers' && model.variant === 'ref2va');

/** components_only folders contain tokenizer/processor/VAEs but require transformer and text-encoder overrides. */
export const isComponentsOnlyVideoMain = (model: MainModelConfig): boolean =>
  // Require full configs so narrowing cannot silently omit components_only.
  (model.base === 'minimax-h3' || model.base === 'ltx-2') &&
  model.format === 'diffusers' &&
  model.components_only === true;

const getVideoVariantConfig = (
  model: Pick<MainModelConfig, 'base' | 'type' | 'variant' | 'format'> | undefined
): VideoVariantConfig | null => {
  if (!model || !isSupportedVideoModel(model)) {
    return null;
  }

  const baseEntry = VIDEO_GENERATION[model.base as SupportedVideoBase];
  const variant = typeof model.variant === 'string' ? model.variant : '';

  return baseEntry.variants[variant] ?? baseEntry.fallback;
};

// Fallback keeps UI selectors crash-safe while nothing is selected; isSupportedVideoModel() still blocks invocation.
const FALLBACK_VARIANT_CONFIG = WAN_FALLBACK_VARIANT;

const getVideoConfig = (
  model: Pick<MainModelConfig, 'base' | 'type' | 'variant' | 'format'> | undefined
): VideoVariantConfig => getVideoVariantConfig(model) ?? FALLBACK_VARIANT_CONFIG;

/** `ltx2_denoise` refuses a refine pass alongside a held-clean conditioning clip. */
export const isTwoStageSupportedForMode = (mode: VideoGenerationMode): boolean =>
  mode !== 'audio-to-video' && mode !== 'video-to-audio';

export const getVideoModes = (model: MainModelConfig | undefined): readonly VideoGenerationMode[] =>
  getVideoConfig(model).modes;

export const isVideoModeSupported = (model: MainModelConfig | undefined, mode: VideoGenerationMode): boolean =>
  getVideoModes(model).includes(mode);

export const getVideoFramesPolicy = (model: MainModelConfig | undefined): VideoFramesPolicy =>
  getVideoConfig(model).frames;

export const getVideoFpsPolicy = (model: MainModelConfig | undefined): VideoFpsPolicy => getVideoConfig(model).fps;

export const getVideoTargetResolutionOptions = (
  model: MainModelConfig | undefined
): readonly VideoTargetResolutionOption[] => getVideoConfig(model).targetResolutions;

export const getVideoAspectRatioOptions = (_model: MainModelConfig | undefined): readonly VideoAspectRatioId[] =>
  VIDEO_ASPECT_RATIO_IDS;

const coerceTargetResolution = (
  config: VideoVariantConfig,
  targetResolution: VideoTargetResolution
): VideoTargetResolution =>
  config.targetResolutions.some((option) => option.id === targetResolution)
    ? targetResolution
    : config.defaults.targetResolution;

/** A record persisted under another family keeps its own preset until a model selection re-coerces it. */
export const getVideoTargetResolution = (
  model: MainModelConfig | undefined,
  targetResolution: VideoTargetResolution
): VideoTargetResolution => coerceTargetResolution(getVideoConfig(model), targetResolution);

export const snapVideoNumFrames = (model: MainModelConfig | undefined, numFrames: number): number => {
  const frames = getVideoConfig(model).frames;

  return frames.kind === 'grid' ? snapNumFramesToGrid(frames, numFrames) : snapNumFramesToChoices(frames, numFrames);
};

/** How a grid family states its rule, e.g. "8·n + 1" — the form the backend node rejects on. */
export const describeVideoFramesGrid = (frames: VideoFramesGridPolicy): string =>
  `${frames.step}·n + ${frames.min - frames.step}`;

export const isValidVideoNumFrames = (model: MainModelConfig | undefined, numFrames: number): boolean => {
  const frames = getVideoConfig(model).frames;

  if (frames.kind === 'choices') {
    return frames.choices.includes(numFrames);
  }

  return (
    Number.isInteger(numFrames) &&
    numFrames >= frames.min &&
    numFrames <= frames.max &&
    (numFrames - frames.min) % frames.step === 0
  );
};

export type VideoDimensionSource = 'aspect-ratio' | 'first-frame' | 'last-frame' | 'source-video' | 'conditioning-clip';

export interface ResolvedVideoDimensions extends VideoDimensions {
  source: VideoDimensionSource;
}

/** Conditioning media overrides the preset aspect ratio; null for unsupported ratios or degenerate inputs. */
export const getVideoDimensions = (
  model: MainModelConfig | undefined,
  settings: Pick<
    VideoSettings,
    | 'aspectRatioId'
    | 'targetResolution'
    | 'firstFrameImage'
    | 'lastFrameImage'
    | 'sourceVideo'
    | 'conditioningClip'
    | 'h3TaskOverride'
  >
): ResolvedVideoDimensions | null => {
  model = getEffectiveVideoModel(model, settings);
  const config = getVideoConfig(model);
  const targetResolution = coerceTargetResolution(config, settings.targetResolution);

  // In the `audio` role the picture is generated, so the aspect-ratio control still owns the frame.
  const conditioningPicture =
    settings.conditioningClip?.role === 'video' && config.modes.includes('video-to-audio')
      ? settings.conditioningClip.clip
      : null;

  const media = conditioningPicture
    ? { ...conditioningPicture, source: 'conditioning-clip' as const }
    : settings.sourceVideo
      ? { ...settings.sourceVideo, source: 'source-video' as const }
      : settings.firstFrameImage
        ? { ...settings.firstFrameImage, source: 'first-frame' as const }
        : settings.lastFrameImage
          ? { ...settings.lastFrameImage, source: 'last-frame' as const }
          : null;

  const inputs = media ?? { ...getVideoAspectRatioParts(settings.aspectRatioId), source: 'aspect-ratio' as const };

  const dimensions =
    model?.base === 'minimax-h3'
      ? resolveMiniMaxH3Canvas(inputs.width, inputs.height, targetResolution as MiniMaxH3TargetResolution)
      : model?.base === 'ltx-2'
        ? resolveLtx2Canvas(inputs.width, inputs.height, targetResolution as Ltx2TargetResolution)
        : scaleAndSnapWanDimensions(
            inputs.width,
            inputs.height,
            targetResolution as '480p' | '720p' | '1080p',
            config.pixelMultiple
          );

  return dimensions ? { ...dimensions, source: inputs.source } : null;
};

/** A clip decides timing for the modality it holds clean; stored values stay so clearing the clip restores them. */
export interface EffectiveVideoTiming {
  numFrames: number;
  fps: number;
  numFramesFromClip: boolean;
  fpsFromClip: boolean;
}

/** A conditioning clip drives `num_frames` itself; an extension's head still sizes the new material. */
export const isAutoDurationSupportedForMode = (
  settings: Pick<
    VideoSettings,
    'conditioningClip' | 'firstFrameImage' | 'lastFrameImage' | 'references' | 'sourceVideo'
  >
): boolean => {
  const mode = resolveVideoMode(settings);

  return mode !== 'audio-to-video' && mode !== 'video-to-audio';
};

export const isAutoDurationActive = (
  settings: Pick<
    VideoSettings,
    | 'autoDuration'
    | 'conditioningClip'
    | 'firstFrameImage'
    | 'lastFrameImage'
    | 'ltx2DurationHeadModel'
    | 'references'
    | 'sourceVideo'
  >
): boolean =>
  Boolean(settings.autoDuration && settings.ltx2DurationHeadModel) && isAutoDurationSupportedForMode(settings);

export const getEffectiveVideoTiming = (
  model: MainModelConfig | undefined,
  settings: Pick<
    VideoSettings,
    'conditioningClip' | 'firstFrameImage' | 'fps' | 'lastFrameImage' | 'numFrames' | 'references' | 'sourceVideo'
  >
): EffectiveVideoTiming => {
  const conditioning = settings.conditioningClip;

  if (!conditioning || !getVideoConfig(model).modes.includes(resolveVideoMode(settings))) {
    return { fps: settings.fps, fpsFromClip: false, numFrames: settings.numFrames, numFramesFromClip: false };
  }

  // The clip's rate only in the `video` role and when the gallery knew it; a generated picture uses the panel's.
  const fpsFromClip = conditioning.role === 'video' && conditioning.fpsKnown;
  const fps = fpsFromClip ? conditioning.clip.fps : settings.fps;

  return {
    fps,
    fpsFromClip,
    numFrames: ltx2FramesForClip(conditioning, fps),
    numFramesFromClip: true,
  };
};

/** A guidance-triple accelerator sends `schedule: 'distilled'`; the backend then discards scales and steps. */
const isGuidanceDistilled = (
  config: VideoVariantConfig,
  settings: Pick<VideoSettings, 'acceleratorEnabled'>
): boolean => settings.acceleratorEnabled && config.accelerator?.guidance !== undefined;

export const getVideoPromptPolicy = (
  model: MainModelConfig | undefined,
  settings: Pick<
    VideoSettings,
    | 'acceleratorEnabled'
    | 'audioCfgScale'
    | 'cfgScale'
    | 'cfgScaleLowNoise'
    | 'negativePromptEnabled'
    | 'wanLowNoiseModel'
  >
) => {
  const config = getVideoConfig(model);
  // Match wan_video_denoise.do_cfg: a second expert using CFG > 1 also consumes negative conditioning.
  const lowNoiseCfgActive =
    config.cfg.lowNoiseVisible &&
    (model?.format === 'diffusers' || settings.wanLowNoiseModel !== null) &&
    settings.cfgScaleLowNoise !== null &&
    settings.cfgScaleLowNoise > 1;
  // LTX-2's audio CFG > 1 shares the unconditional pass, so it consumes the negative prompt even at video CFG 1.
  const audioCfgActive = config.guidance.audioVisible && settings.audioCfgScale !== null && settings.audioCfgScale > 1;
  // A guidance-distilled run presents like the distilled checkpoint (`usage: 'never'`).
  const guidanceDistilled = isGuidanceDistilled(config, settings);
  const negativeUsedInGraph =
    !guidanceDistilled &&
    settings.negativePromptEnabled &&
    (config.negativePrompt.usage === 'always' ||
      (config.negativePrompt.usage === 'cfg-gated' && (settings.cfgScale > 1 || lowNoiseCfgActive || audioCfgActive)));

  return {
    enhancer: config.promptEnhancer ?? null,
    negativeVisible: config.negativePrompt.visible && !guidanceDistilled,
    negativeUsedInGraph,
    ...(config.negativePrompt.usage === 'cfg-gated' ? { negativeHelpTextKey: 'widgets.video.negativeCfgHelp' } : {}),
  };
};

/** Only a first frame is offered: it opens the video the prompt describes. */
export const getVideoExpandPromptSuggestion = (
  enhancer: VideoPromptEnhancer | null,
  firstFrameImage: ImageWithDims | null
): ExpandPromptSuggestion | null =>
  enhancer
    ? {
        image: firstFrameImage,
        imageSystemPromptId: enhancer.imageToVideoSystemPromptId,
        modelName: enhancer.modelName,
        modelSource: enhancer.modelSource,
        systemPromptId: enhancer.textToVideoSystemPromptId,
      }
    : null;

export interface VideoModelPolicy {
  isSupported: boolean;
  modes: readonly VideoGenerationMode[];
  pixelMultiple: number;
  minSteps: number;
  aspectRatioOptions: readonly VideoAspectRatioId[];
  targetResolutions: readonly VideoTargetResolutionOption[];
  refineSteps?: number;
  frames: VideoFramesPolicy;
  fps: VideoFpsPolicy;
  defaults: {
    targetResolution: VideoTargetResolution;
    steps: number;
    cfgScale: number;
    cfgScaleLowNoise: number | null;
    audioCfgScale: number | null;
    stgScale: number | null;
    modalityScale: number | null;
  };
  prompt: {
    enhancer: VideoPromptEnhancer | null;
    negativeVisible: boolean;
    negativeUsedInGraph: boolean;
    negativeHelpTextKey?: string;
  };
  references: { maxVideos: number; maxImages: number; extend?: boolean } | null;
  ui: {
    cfgVisible: boolean;
    cfgLowNoiseVisible: boolean;
    /** LTX-2's per-modality guidance controls; all false on the other families. */
    audioCfgVisible: boolean;
    stgVisible: boolean;
    modalityVisible: boolean;
    fpsVisible: boolean;
    stepsEditable: boolean;
    /** `newFrames` is what the run adds once the crossfade consumes the context from both halves. */
    extendContext: { value: number; min: number; max: number; step: number; newFrames: number } | null;
    accelerator: VideoAcceleratorConfig | null;
    acceleratorSteps: number | null;
    audioOutput: boolean;
  };
}

export interface AutoDurationBounds {
  maxSeconds: number;
  minSeconds: number;
  /** Frames held in front of the prediction: an extension's context, else 0. */
  contextFrames: number;
  /** The longest total the head can hand the run, context included, at `fps`. */
  maxFrames: number;
  maxNewFrames: number;
  /** The rate the seconds are read at; a continuation's is the source's. */
  fps: number;
}

/** The range the duration head picks from, or null when it does not run; graph and panel both read this. */
export const getAutoDurationBounds = (
  model: MainModelConfig | undefined,
  settings: VideoSettings
): AutoDurationBounds | null => {
  if (!isAutoDurationActive(settings)) {
    return null;
  }
  const frames = getVideoModelPolicy(model, settings).frames;
  if (frames.kind !== 'grid') {
    return null;
  }
  const timing = getEffectiveVideoTiming(model, settings);
  const source = resolveVideoMode(settings) === 'extend' ? settings.sourceVideo : null;
  const fps = source ? source.fps : timing.fps;
  const contextFrames = source ? settings.ltx2ExtendContextFrames : 0;
  // Both 8k + 1, so their difference plus one is too: the longest prediction that still fits.
  const bounds = ltx2AutoDurationBounds(fps, frames.min, timing.numFrames - Math.max(0, contextFrames - 1));
  if (!bounds) {
    return null;
  }
  // The head converts seconds at the run's rate and floors onto the grid, the same as here.
  const predicted = snapLtx2FramesDown(Math.round(bounds.maxSeconds * fps));

  return {
    ...bounds,
    contextFrames,
    fps,
    maxFrames: contextFrames ? contextFrames + predicted - 1 : predicted,
    maxNewFrames: contextFrames ? predicted - 1 : predicted,
  };
};

export const getVideoModelPolicy = (model: MainModelConfig | undefined, settings: VideoSettings): VideoModelPolicy => {
  model = getEffectiveVideoModel(model, settings);
  const config = getVideoConfig(model);
  const guidanceDistilled = isGuidanceDistilled(config, settings);
  // Only for LTX-2 extend with a source picked: the ceiling is the source's.
  const extendContext =
    model?.base === 'ltx-2' && resolveVideoMode(settings) === 'extend' && settings.sourceVideo
      ? {
          value: settings.ltx2ExtendContextFrames,
          min: 1 + LTX2_NUM_FRAMES_STEP,
          max: ltx2MaxExtendContextFrames(settings.sourceVideo.width, settings.sourceVideo.height),
          step: LTX2_NUM_FRAMES_STEP,
          newFrames: ltx2NewFramesForExtend(settings.numFrames, settings.ltx2ExtendContextFrames),
        }
      : null;

  return {
    aspectRatioOptions: getVideoAspectRatioOptions(model),
    defaults: config.defaults,
    fps: config.fps,
    frames: config.frames,
    isSupported: model ? isSupportedVideoModel(model) : false,
    minSteps: config.minSteps,
    modes: config.modes,
    pixelMultiple: config.pixelMultiple,
    prompt: getVideoPromptPolicy(model, settings),
    references: config.references ?? null,
    ...(config.refineSteps === undefined ? {} : { refineSteps: config.refineSteps }),
    targetResolutions: config.targetResolutions,
    ui: {
      accelerator: config.accelerator,
      audioCfgVisible: config.guidance.audioVisible && !guidanceDistilled,
      // Show the active LoRA's schedule while retaining accelerator as the family reference config.
      acceleratorSteps: config.accelerator
        ? getAcceleratorSteps(config.accelerator, getRecordedAcceleratorLoras(settings))
        : null,
      audioOutput: config.audioOutput,
      extendContext: extendContext,
      cfgLowNoiseVisible: config.cfg.lowNoiseVisible && !guidanceDistilled,
      cfgVisible: config.cfg.visible && !guidanceDistilled,
      fpsVisible: config.fps.editable,
      modalityVisible: config.guidance.modalityVisible && !guidanceDistilled,
      stepsEditable: config.stepsEditable && !guidanceDistilled,
      stgVisible: config.guidance.stgVisible && !guidanceDistilled,
    },
  };
};

export interface WanLightningLoraPair {
  high: LoraModelConfig;
  low: LoraModelConfig;
}

export interface FindAcceleratorLoraOptions {
  /** Require the model-family token for user-list candidates to reject look-alike or wrong-family accelerators. */
  requireFamilyName?: boolean;
}

export interface FindMiniMaxH3TurboLoraOptions extends FindAcceleratorLoraOptions {
  /** The effective model's variant; ref2va REQUIRES the ref2v-token repack, every other variant excludes it. */
  variant?: string | null;
}

// Non-alphanumeric delimiters: \b misses underscore tokens, and suffix matching accepts Slow/Thigh.
const HIGH_NOISE_PATTERN = /(?:^|[^a-z0-9])high(?:[^a-z0-9]|$)/i;
const LOW_NOISE_PATTERN = /(?:^|[^a-z0-9])low(?:[^a-z0-9]|$)/i;

/** Prefer the main's family; requireFamilyName rejects fallback matches. */
export const findWanLightningLoraPair = (
  models: readonly ModelConfig[],
  mainVariant: string | null | undefined,
  { requireFamilyName = false }: FindAcceleratorLoraOptions = {}
): WanLightningLoraPair | null => {
  const candidates = models.filter(
    (model): model is ModelConfig & LoraModelConfig =>
      isLoraModelConfig(model) &&
      model.base === 'wan' &&
      /lightning/i.test(model.name) &&
      isWanLoraTargetingMain(model.variant, mainVariant)
  );

  // Delimited-token match so 'i2v' cannot score inside "TI2V".
  const familyToken = typeof mainVariant === 'string' ? mainVariant.split('_')[0] : undefined;
  const familyPattern = familyToken ? new RegExp(`(?:^|[^a-z0-9])${familyToken}(?:[^a-z0-9]|$)`, 'i') : null;
  const score = (model: LoraModelConfig): number => (familyPattern?.test(model.name) ? 0 : 1);
  const pick = (pattern: RegExp): LoraModelConfig | null =>
    candidates
      .filter((model) => pattern.test(model.name) && (!requireFamilyName || score(model) === 0))
      .sort((a, b) => score(a) - score(b))[0] ?? null;

  const high = pick(HIGH_NOISE_PATTERN);
  const low = pick(LOW_NOISE_PATTERN);

  return high && low && high.key !== low.key ? { high, low } : null;
};

const TURBO_PATTERN = /(?:^|[^a-z0-9])turbo(?:[^a-z0-9]|$)/i;
// Recognize both starter names and raw LightX2V filenames retained by URL installs.
const LIGHTX2V_RELEASE_PATTERN =
  /(?:^|[^a-z0-9])lightx2v(?:[^a-z0-9]|$)|minimax_h3_(?:fl2v|ref2v)_turbo_\d{1,2}step_v\d/i;
const MINIMAX_H3_NAME_PATTERN = /(?:^|[^a-z0-9])(?:minimax|h3)(?:[^a-z0-9]|$)/i;
// Ref2V Turbo is Ref2VA-trained and must never auto-apply to FL2VA.
const MINIMAX_H3_REF2V_PATTERN = /(?:^|[^a-z0-9])ref2va?(?:[^a-z0-9]|$)/i;

export const findMiniMaxH3TurboLora = (
  models: readonly ModelConfig[],
  { requireFamilyName = false, variant = 'fl2va' }: FindMiniMaxH3TurboLoraOptions = {}
): LoraModelConfig | null => {
  // Family name first (0/2), then the LightX2V generation (0/1).
  const score = (model: LoraModelConfig): number =>
    (MINIMAX_H3_NAME_PATTERN.test(model.name) ? 0 : 2) + (LIGHTX2V_RELEASE_PATTERN.test(model.name) ? 0 : 1);
  // Higher stated schedules rank newer. Ref2VA requires Ref2V-trained releases; other tasks exclude them.
  const matchesTask = (name: string): boolean =>
    variant === 'ref2va' ? MINIMAX_H3_REF2V_PATTERN.test(name) : !MINIMAX_H3_REF2V_PATTERN.test(name);

  return (
    models
      .filter(
        (model): model is ModelConfig & LoraModelConfig =>
          isLoraModelConfig(model) &&
          model.base === 'minimax-h3' &&
          TURBO_PATTERN.test(model.name) &&
          matchesTask(model.name) &&
          (!requireFamilyName || MINIMAX_H3_NAME_PATTERN.test(model.name))
      )
      .sort(
        (a, b) =>
          score(a) - score(b) || getStatedStepCount(b.name) - getStatedStepCount(a.name) || a.name.localeCompare(b.name)
      )[0] ?? null
  );
};

/** The release name states its rank, not a schedule (`…-distilled-lora-450`), so match on "distilled". */
const LTX2_DISTILLED_PATTERN = /(?:^|[^a-z0-9])distill(?:ed|ation)?(?:[^a-z0-9]|$)/i;
const LTX2_NAME_PATTERN = /(?:^|[^a-z0-9])ltx[ _-]?2(?:\.\d)?(?:[^a-z0-9]|$)/i;

export const findLtx2DistilledLora = (
  models: readonly ModelConfig[],
  { requireFamilyName = false }: FindAcceleratorLoraOptions = {}
): LoraModelConfig | null => {
  const matches = models.filter(
    (model): model is LoraModelConfig =>
      isLoraModelConfig(model) &&
      model.base === 'ltx-2' &&
      LTX2_DISTILLED_PATTERN.test(model.name) &&
      (!requireFamilyName || LTX2_NAME_PATTERN.test(model.name))
  );

  return (
    matches
      .slice()
      // Family-named first, then a deterministic tie-break — there is no step count to rank on.
      .sort(
        (a, b) =>
          Number(LTX2_NAME_PATTERN.test(b.name)) - Number(LTX2_NAME_PATTERN.test(a.name)) ||
          a.name.localeCompare(b.name)
      )[0] ?? null
  );
};

/** A complete accelerator set or null. Pass the effective model: its H3 task variant picks the Turbo release. */
export const findAcceleratorLorasIn = (
  model: MainModelConfig,
  candidates: readonly ModelConfig[],
  options: FindAcceleratorLoraOptions = {}
): LoraModelConfig[] | null => {
  const find = ACCELERATOR_FINDERS[model.base as SupportedVideoBase];

  return find ? find(candidates, model, options) : null;
};

/** Wan's accelerator is a pair, one LoRA per expert; the rest are single. */
const ACCELERATOR_FINDERS: Record<
  SupportedVideoBase,
  (
    candidates: readonly ModelConfig[],
    model: MainModelConfig,
    options: FindAcceleratorLoraOptions
  ) => LoraModelConfig[] | null
> = {
  'ltx-2': (candidates, _model, options) => {
    const distilled = findLtx2DistilledLora(candidates, options);

    return distilled ? [distilled] : null;
  },
  'minimax-h3': (candidates, model, options) => {
    const turbo = findMiniMaxH3TurboLora(candidates, {
      ...options,
      variant: typeof model.variant === 'string' ? model.variant : 'fl2va',
    });

    return turbo ? [turbo] : null;
  },
  wan: (candidates, model, options) => {
    const pair = findWanLightningLoraPair(candidates, model.variant, options);

    return pair ? [pair.high, pair.low] : null;
  },
};

// Limit schedule tokens to two digits so checkpoint tags such as step600 cannot become step counts.
const STEP_COUNT_PATTERN = /(?:^|[^a-z0-9])(\d{1,2})[ _-]?steps?(?:[^a-z0-9]|$)/i;

/** The schedule a distillation LoRA's name states, or 0 when it states none. */
const getStatedStepCount = (name: string): number => Number(STEP_COUNT_PATTERN.exec(name)?.[1] ?? 0);

const getLoraAcceleratorSteps = (config: VideoAcceleratorConfig, lora: LoraModelConfig): number => {
  const stated = getStatedStepCount(lora.name);

  if (stated > 0) {
    return stated;
  }

  return config.stepOverrides?.find((override) => override.pattern.test(lora.name))?.steps ?? config.steps;
};

/** Use the higher count for mismatched expert schedules to avoid under-stepping. */
export const getAcceleratorSteps = (config: VideoAcceleratorConfig, loras: readonly LoraModelConfig[]): number =>
  loras.reduce((steps, lora) => Math.max(steps, getLoraAcceleratorSteps(config, lora)), 0) || config.steps;

const getEnabledLoraModels = (settings: Pick<VideoSettings, 'loras'>): LoraModelConfig[] =>
  settings.loras.filter((lora) => lora.isEnabled).map((lora) => lora.model);

const getRecordedAcceleratorLoras = (
  settings: Pick<VideoSettings, 'acceleratorEnabled' | 'acceleratorLoraKeys' | 'loras'>
): LoraModelConfig[] => {
  const recorded = new Set(settings.acceleratorLoraKeys);

  return settings.acceleratorEnabled
    ? settings.loras.filter((lora) => recorded.has(lora.model.key)).map((lora) => lora.model)
    : [];
};

/** Complete family matches, or the exact recorded set (which may hold unusual names the toggle installed). */
const findAcceleratorAmong = (
  model: MainModelConfig,
  candidates: readonly LoraModelConfig[],
  models: readonly ModelConfig[]
): LoraModelConfig[] | null => {
  const named = findAcceleratorLorasIn(model, candidates, { requireFamilyName: true });

  if (named) {
    return named;
  }

  const catalogPick = findAcceleratorLorasIn(model, models);

  return catalogPick?.every((lora) => candidates.some((candidate) => candidate.key === lora.key)) ? catalogPick : null;
};

/** Validate only recorded keys, so another enabled candidate cannot displace the user's anchored set via tie-breaks. */
const isRecordedAcceleratorIntact = (
  settings: Pick<VideoSettings, 'acceleratorLoraKeys' | 'loras'>,
  model: MainModelConfig,
  models: readonly ModelConfig[],
  config: VideoVariantConfig
): boolean => {
  const recorded = new Set(settings.acceleratorLoraKeys);

  if (!config.accelerator || recorded.size === 0) {
    return false;
  }

  // Deduplicate keys so repeated persisted LoRAs can still reach the unchanged reconciliation outcome.
  const live = [
    ...new Map(
      getEnabledLoraModels(settings)
        .filter((lora) => recorded.has(lora.key))
        .map((lora) => [lora.key, lora])
    ).values(),
  ];

  return live.length === recorded.size && findAcceleratorAmong(model, live, models)?.length === recorded.size;
};

/** Choose only from the installed catalog: user-list names cannot tell distillation from personal style LoRAs. */
const findAcceleratorLoraEntries = (model: MainModelConfig, models: readonly ModelConfig[]): GenerateLora[] | null =>
  findAcceleratorLorasIn(model, models)?.map((lora) => ({ isEnabled: true, model: lora, weight: 1 })) ?? null;

export interface AcceleratorToggleResult {
  settings: VideoSettings;
  missingLoras: boolean;
}

/** Encode acceleration through visible LoRAs and sampling settings, so graph compilation has no hidden toggle state. */
export const getAcceleratorToggleResult = (
  settings: VideoSettings,
  model: MainModelConfig,
  models: readonly ModelConfig[],
  enabled: boolean
): AcceleratorToggleResult => {
  model = getEffectiveVideoModel(model, settings);
  const config = getVideoConfig(model);
  // Remove only the entries the previous toggle added, never a user's own Lightning-style LoRA.
  const previousKeys = new Set(settings.acceleratorLoraKeys);
  const withoutAccelerators = settings.loras.filter((lora) => !previousKeys.has(lora.model.key));

  if (!enabled || !config.accelerator) {
    return {
      missingLoras: false,
      settings: {
        ...settings,
        acceleratorEnabled: false,
        acceleratorLoraKeys: [],
        cfgScale: config.defaults.cfgScale,
        cfgScaleLowNoise: config.defaults.cfgScaleLowNoise,
        // Only where the accelerator lowered them; otherwise this would reset the user's own guidance tweak.
        ...(config.accelerator?.guidance
          ? {
              audioCfgScale: config.defaults.audioCfgScale,
              modalityScale: config.defaults.modalityScale,
              stgScale: config.defaults.stgScale,
            }
          : {}),
        loras: withoutAccelerators,
        steps: config.defaults.steps,
      },
    };
  }

  const entries = findAcceleratorLoraEntries(model, models);

  if (!entries) {
    // Never leave the flag claiming a fast path that has no LoRAs behind it.
    return {
      missingLoras: true,
      settings: settings.acceleratorEnabled
        ? { ...settings, acceleratorEnabled: false, acceleratorLoraKeys: [] }
        : settings,
    };
  }

  return {
    missingLoras: false,
    settings: {
      ...settings,
      acceleratorEnabled: true,
      acceleratorLoraKeys: entries.map((entry) => entry.model.key),
      cfgScale: config.accelerator.cfgScale,
      cfgScaleLowNoise: config.accelerator.cfgScaleLowNoise,
      ...config.accelerator.guidance,
      loras: [
        ...withoutAccelerators.filter((lora) => !entries.some((e) => e.model.key === lora.model.key)),
        ...entries,
      ],
      steps: getAcceleratorSteps(
        config.accelerator,
        entries.map((entry) => entry.model)
      ),
    },
  };
};

/** Switch H3 tasks without leaving Turbo's sampling recipe behind when its adapter is removed. */
export const getH3ModeSwitchSettings = (
  settings: VideoSettings,
  model: MainModelConfig,
  models: readonly ModelConfig[],
  override: boolean
): VideoSettings => {
  const detected = getEffectiveVideoModel(model, {}).variant;
  const h3TaskOverride = override ? (detected === 'ref2va' ? 'fl2va' : 'ref2va') : undefined;
  const next: VideoSettings = { ...settings, h3TaskOverride };
  return settings.acceleratorEnabled ? getAcceleratorToggleResult(next, model, models, false).settings : next;
};

export type AcceleratorLoraChangeOutcome = 'unchanged' | 'switched' | 'disabled';

export interface AcceleratorLoraChangeResult {
  settings: VideoSettings;
  outcome: AcceleratorLoraChangeOutcome;
  /** The LoRA set now driving the fast path — only set when `outcome` is 'switched'. */
  acceleratorLoras: LoraModelConfig[] | null;
}

/** Never enables from a list edit: name heuristics cannot prove a user LoRA supports distillation. */
export const getAcceleratorLoraChangeResult = (
  settings: VideoSettings,
  model: MainModelConfig,
  models: readonly ModelConfig[],
  loras: GenerateLora[]
): AcceleratorLoraChangeResult => {
  model = getEffectiveVideoModel(model, settings);
  const config = getVideoConfig(model);
  const next: VideoSettings = { ...settings, loras };

  if (!settings.acceleratorEnabled) {
    return { acceleratorLoras: null, outcome: 'unchanged', settings: next };
  }

  if (isRecordedAcceleratorIntact(next, model, models, config)) {
    return { acceleratorLoras: null, outcome: 'unchanged', settings: next };
  }

  const replacement = config.accelerator ? findAcceleratorAmong(model, getEnabledLoraModels(next), models) : null;

  if (replacement && config.accelerator) {
    return {
      acceleratorLoras: replacement,
      outcome: 'switched',
      settings: {
        ...next,
        acceleratorEnabled: true,
        acceleratorLoraKeys: replacement.map((lora) => lora.key),
        cfgScale: config.accelerator.cfgScale,
        cfgScaleLowNoise: config.accelerator.cfgScaleLowNoise,
        ...config.accelerator.guidance,
        steps: getAcceleratorSteps(config.accelerator, replacement),
      },
    };
  }

  return {
    acceleratorLoras: null,
    outcome: 'disabled',
    settings: {
      ...next,
      acceleratorEnabled: false,
      acceleratorLoraKeys: [],
      cfgScale: config.defaults.cfgScale,
      cfgScaleLowNoise: config.defaults.cfgScaleLowNoise,
      // Without this, a guided run keeps the accelerator's identity scales: most of its guidance silently off.
      ...(config.accelerator?.guidance
        ? {
            audioCfgScale: config.defaults.audioCfgScale,
            modalityScale: config.defaults.modalityScale,
            stgScale: config.defaults.stgScale,
          }
        : {}),
      steps: config.defaults.steps,
    },
  };
};

export type VideoComponentValueKey =
  | 'vae'
  | 'wanT5EncoderModel'
  | 'wanLowNoiseModel'
  | 'componentSourceModel'
  | 'h3TransformerModel'
  | 'h3TextEncoderModel'
  | 'h3HybridBaseModel'
  | 'ltx2TextEncoderModel'
  | 'ltx2DurationHeadModel';

export interface VideoComponentPolicyContext {
  model: MainModelConfig;
  settings: VideoSettings;
  selectedComponents: Pick<VideoSettings, VideoComponentValueKey>;
}

export interface VideoComponentSlotPolicy {
  key: VideoComponentValueKey;
  label: string;
  modelTypes: readonly ModelTaxonomyType[];
  valueKind: 'component' | 'vae' | 'main';
  /** Translation key for the slot's inline help. The core is UI-free, so the section resolves it. */
  helpTextKey?: string;
  filter?: (candidate: ModelConfig, ctx: VideoComponentPolicyContext) => boolean;
  required?: (ctx: VideoComponentPolicyContext) => boolean;
  missingMessage?: string;
}

export interface VideoComponentSectionPolicy {
  defaultOpen: boolean;
  slots: readonly VideoComponentSlotPolicy[];
  validate: (ctx: VideoComponentPolicyContext) => string[];
}

const VIDEO_COMPONENT_SETTING_LABELS: Record<VideoComponentValueKey, string> = {
  componentSourceModel: 'Component source',
  ltx2DurationHeadModel: 'Duration head',
  ltx2TextEncoderModel: 'Gemma-4 encoder',
  h3HybridBaseModel: 'Hybrid quality base',
  h3TextEncoderModel: 'Text encoder (single file)',
  h3TransformerModel: 'Transformer (single file)',
  vae: 'VAE',
  wanLowNoiseModel: 'Low-noise expert',
  wanT5EncoderModel: 'Wan T5 Encoder',
};

const isTi2v5b = (variant: unknown): boolean => variant === 'ti2v_5b';

// Diffusers Wan mains bundle VAE and encoder; GGUF/checkpoint mains need standalone models or a component source.
const getWanComponentSource = (ctx: VideoComponentPolicyContext) =>
  getCompatibleDiffusersComponentSource(ctx.model, ctx.settings.componentSourceModel);

// The UMT5-XXL encoder is shared across Wan families, so any Diffusers source supplies it.
const isWanEncoderSatisfied = (ctx: VideoComponentPolicyContext): boolean =>
  ctx.model.format === 'diffusers' || Boolean(getWanComponentSource(ctx));

// Family-bound (wan_model_loader validation): a source covers it only when its TI2V-5B-ness matches.
const isWanVaeSatisfied = (ctx: VideoComponentPolicyContext): boolean => {
  if (ctx.model.format === 'diffusers') {
    return true;
  }

  const source = getWanComponentSource(ctx);

  return Boolean(source) && isTi2v5b(source?.variant) === isTi2v5b(ctx.model.variant);
};

const validateSlots = (slots: readonly VideoComponentSlotPolicy[], ctx: VideoComponentPolicyContext): string[] =>
  slots.flatMap((slotPolicy) => {
    if (!slotPolicy.required?.(ctx)) {
      return [];
    }

    const value = ctx.selectedComponents[slotPolicy.key];
    const isValid = value && (!slotPolicy.filter || slotPolicy.filter(value as ModelConfig, ctx));

    return isValid ? [] : [slotPolicy.missingMessage ?? `Video needs a ${slotPolicy.label} for this model.`];
  });

const createComponentPolicy = (
  defaultOpen: boolean,
  slots: readonly VideoComponentSlotPolicy[]
): VideoComponentSectionPolicy => ({
  defaultOpen,
  slots,
  validate: (ctx) => validateSlots(slots, ctx),
});

const EMPTY_VIDEO_COMPONENT_POLICY = createComponentPolicy(false, []);

// Distinct single-file models of the same variant; unknown variants are left to backend probing.
const isWanLowNoiseExpertCandidate = (candidate: ModelConfig, ctx: VideoComponentPolicyContext): boolean =>
  candidate.type === 'main' &&
  candidate.base === 'wan' &&
  candidate.format !== 'diffusers' &&
  candidate.key !== ctx.model.key &&
  (typeof candidate.variant !== 'string' ||
    typeof ctx.model.variant !== 'string' ||
    candidate.variant === ctx.model.variant);

// TI2V-5B needs 48 VAE channels, A14B 16. Not awaiting capabilities: early sync could drop a valid stored VAE.
const isWanVaeForMain = (candidate: ModelConfig, ctx: VideoComponentPolicyContext): boolean => {
  if (candidate.type !== 'vae' || candidate.base !== 'wan') {
    return false;
  }

  const latentChannels = candidate.latent_channels;

  if (typeof latentChannels !== 'number') {
    return true;
  }

  return latentChannels === (isTi2v5b(ctx.model.variant) ? 48 : 16);
};

// Both spelled out as `…Key` properties so the translation-key scan sees them.
const H3_DIFFUSERS_TEXT_ENCODER_HELP = {
  optionalKey: 'widgets.video.componentSlots.h3TextEncoderOptionalHelp',
  requiredKey: 'widgets.video.componentSlots.h3TextEncoderRequiredHelp',
} as const;

export const getVideoComponentSectionPolicy = (
  model: MainModelConfig | undefined,
  _settings: VideoSettings
): VideoComponentSectionPolicy => {
  if (!model || !isSupportedVideoModel(model)) {
    return EMPTY_VIDEO_COMPONENT_POLICY;
  }

  if (model.base === 'wan') {
    const config = getVideoConfig(model);
    const slots: VideoComponentSlotPolicy[] = [
      // Diffusers mains supply their own components and ignore this input; offer it only for single-file mains.
      ...(model.format === 'diffusers'
        ? []
        : [
            {
              filter: (candidate: ModelConfig, ctx: VideoComponentPolicyContext) =>
                isDiffusersMainForBase('wan')(candidate) && candidate.key !== ctx.model.key,
              helpTextKey: 'widgets.video.componentSlots.wanComponentSourceHelp',
              key: 'componentSourceModel',
              label: 'Component source',
              modelTypes: ['main'],
              valueKind: 'main',
            } satisfies VideoComponentSlotPolicy,
          ]),
      {
        filter: isWanVaeForMain,
        helpTextKey: 'widgets.video.componentSlots.wanOptionalWithSourceHelp',
        key: 'vae',
        label: 'VAE',
        missingMessage: 'Video needs a VAE for Wan models.',
        modelTypes: ['vae'],
        required: (ctx) => !isWanVaeSatisfied(ctx),
        valueKind: 'vae',
      },
      {
        filter: (candidate) => candidate.type === 'wan_t5_encoder',
        helpTextKey: 'widgets.video.componentSlots.wanOptionalWithSourceHelp',
        key: 'wanT5EncoderModel',
        label: 'Wan T5 Encoder',
        missingMessage: 'Video needs a Wan T5 Encoder for Wan models.',
        modelTypes: ['wan_t5_encoder'],
        required: (ctx) => !isWanEncoderSatisfied(ctx),
        valueKind: 'component',
      },
    ];

    // TI2V-5B has one expert; Diffusers A14B bundles the second. Only single-file A14B needs this slot.
    if (config.cfg.lowNoiseVisible && model.format !== 'diffusers') {
      slots.push({
        filter: isWanLowNoiseExpertCandidate,
        helpTextKey: 'widgets.video.componentSlots.wanLowNoiseHelp',
        key: 'wanLowNoiseModel',
        label: 'Transformer (Low Noise)',
        modelTypes: ['main'],
        valueKind: 'main',
      });
    }

    return createComponentPolicy(model.format !== 'diffusers', slots);
  }

  // No LTX-2 main carries the Gemma-4 encoder, so its slot is required rather than an override.
  if (model.base === 'ltx-2') {
    const needsComponentSource = model.format !== 'diffusers' || isComponentsOnlyVideoMain(model);

    return createComponentPolicy(needsComponentSource, [
      ...(needsComponentSource
        ? [
            {
              filter: (candidate: ModelConfig, ctx: VideoComponentPolicyContext) =>
                isDiffusersMainForBase('ltx-2')(candidate) && candidate.key !== ctx.model.key,
              helpTextKey: 'widgets.video.componentSlots.ltx2ComponentSourceHelp',
              key: 'componentSourceModel',
              label: 'Model components',
              missingMessage: `${model.name} is a single-file transformer — select an LTX-2 components install under Model Components.`,
              modelTypes: ['main'],
              required: () => true,
              valueKind: 'main',
            } satisfies VideoComponentSlotPolicy,
          ]
        : []),
      {
        filter: (candidate) => candidate.type === 'gemma4_encoder' && candidate.base === 'ltx-2',
        helpTextKey: 'widgets.video.componentSlots.ltx2TextEncoderHelp',
        key: 'ltx2TextEncoderModel',
        label: 'Gemma-4 encoder',
        missingMessage: 'LTX-2 needs its Gemma-4 text encoder — no LTX-2 model carries one.',
        modelTypes: ['gemma4_encoder'],
        required: () => true,
        valueKind: 'component',
      },
      {
        filter: (candidate) => candidate.type === 'ltx2_duration_head' && candidate.base === 'ltx-2',
        helpTextKey: 'widgets.video.componentSlots.ltx2DurationHeadHelp',
        key: 'ltx2DurationHeadModel',
        label: 'Duration head',
        // Optional, so no `missingMessage`: without a head a run uses the frame count the user set.
        modelTypes: ['ltx2_duration_head'],
        required: () => false,
        valueKind: 'component',
      },
    ]);
  }

  // H3 checkpoint mains need a Diffusers component source; components-only ones also need a Qwen3-VL override.
  if (model.format === 'diffusers') {
    // Legacy components-only mains stay visible but require the missing encoder.
    const componentsOnly = isComponentsOnlyVideoMain(model);

    return createComponentPolicy(componentsOnly, [
      {
        filter: (candidate) => candidate.type === 'qwen3_vl_encoder' && candidate.base === 'minimax-h3',
        helpTextKey: componentsOnly
          ? H3_DIFFUSERS_TEXT_ENCODER_HELP.requiredKey
          : H3_DIFFUSERS_TEXT_ENCODER_HELP.optionalKey,
        key: 'h3TextEncoderModel',
        label: 'Text encoder (single file)',
        missingMessage: `${model.name} is a components-only install — select a single-file Text encoder.`,
        modelTypes: ['qwen3_vl_encoder'],
        required: componentsOnly ? () => true : undefined,
        valueKind: 'component',
      },
    ]);
  }

  return createComponentPolicy(true, [
    {
      filter: (candidate) =>
        candidate.type === 'main' && candidate.base === 'minimax-h3' && candidate.format === 'diffusers',
      helpTextKey: 'widgets.video.componentSlots.h3ComponentSourceHelp',
      key: 'componentSourceModel',
      label: 'Model components',
      missingMessage: `${model.name} is a single-file transformer — select a Diffusers MiniMax H3 install under Model Components.`,
      modelTypes: ['main'],
      required: () => true,
      valueKind: 'main',
    },
    {
      filter: (candidate) => candidate.type === 'qwen3_vl_encoder' && candidate.base === 'minimax-h3',
      helpTextKey: 'widgets.video.componentSlots.h3TextEncoderCheckpointHelp',
      key: 'h3TextEncoderModel',
      label: 'Text encoder (single file)',
      missingMessage:
        'The selected Model Components install has no text-encoder weights — select a single-file Text encoder.',
      modelTypes: ['qwen3_vl_encoder'],
      required: (ctx) => !isH3TextEncoderSatisfied(ctx),
      valueKind: 'component',
    },
    // The hybrid replaces non-AdaLN weights with an FL2VA base while retaining Ref2VA task identity and
    // projections.
    ...(model.variant === 'ref2va' && getEffectiveVideoModel(model, _settings).variant === 'ref2va'
      ? [
          {
            filter: isH3HybridBaseCandidate,
            helpTextKey: 'widgets.video.componentSlots.h3HybridBaseHelp',
            key: 'h3HybridBaseModel',
            label: 'Hybrid quality base (FL2VA)',
            modelTypes: ['main'],
            valueKind: 'main',
          } satisfies VideoComponentSlotPolicy,
        ]
      : []),
  ]);
};

// Pruned/full AdaLN shapes differ: require matching checkpoint kinds, leaving unspecified flags to backend validation.
const isH3HybridBaseCandidate = (candidate: ModelConfig, ctx: VideoComponentPolicyContext): boolean =>
  candidate.type === 'main' &&
  candidate.base === 'minimax-h3' &&
  candidate.format === 'checkpoint' &&
  candidate.variant === 'fl2va' &&
  candidate.key !== ctx.model.key &&
  (typeof candidate.pruned !== 'boolean' ||
    typeof ctx.model.pruned !== 'boolean' ||
    candidate.pruned === ctx.model.pruned);

const getH3ComponentSource = (ctx: VideoComponentPolicyContext): MainModelConfig | null => {
  const source = ctx.settings.componentSourceModel;

  return source && source.base === 'minimax-h3' && source.format === 'diffusers' ? source : null;
};

// Only a full Diffusers source carries the text encoder; otherwise the Qwen3-VL override is required.
const isH3TextEncoderSatisfied = (ctx: VideoComponentPolicyContext): boolean => {
  const source = getH3ComponentSource(ctx);

  return source !== null && !isComponentsOnlyVideoMain(source);
};

const getVideoComponentPolicyContext = (
  model: MainModelConfig,
  settings: VideoSettings
): VideoComponentPolicyContext => ({
  model,
  selectedComponents: {
    componentSourceModel: settings.componentSourceModel,
    h3HybridBaseModel: settings.h3HybridBaseModel,
    ltx2TextEncoderModel: settings.ltx2TextEncoderModel,
    ltx2DurationHeadModel: settings.ltx2DurationHeadModel,
    h3TextEncoderModel: settings.h3TextEncoderModel,
    h3TransformerModel: settings.h3TransformerModel,
    vae: settings.vae,
    wanLowNoiseModel: settings.wanLowNoiseModel,
    wanT5EncoderModel: settings.wanT5EncoderModel,
  },
  settings,
});

export type WanExpertWiringWarning =
  | { kind: 'swapped' }
  | { kind: 'high-as-low' }
  | { kind: 'low-as-main' }
  | { kind: 'single-low' }
  | null;

const getWanExpertTag = (model: MainModelConfig | null): 'high' | 'low' | 'none' => {
  const expert = (model as Record<string, unknown> | null)?.expert;

  return expert === 'high' || expert === 'low' ? expert : 'none';
};

/** Expert tags are advisory filename heuristics; explicit wiring stays authoritative and mismatches never block. */
export const getWanExpertWiringWarning = (
  model: MainModelConfig | null,
  wanLowNoiseModel: MainModelConfig | null
): WanExpertWiringWarning => {
  if (!model || model.base !== 'wan' || model.format === 'diffusers') {
    return null;
  }

  const config = getVideoConfig(model);

  if (!config.cfg.lowNoiseVisible) {
    return null;
  }

  const mainTag = getWanExpertTag(model);
  const lowTag = getWanExpertTag(wanLowNoiseModel);

  if (wanLowNoiseModel) {
    if (mainTag === 'low' && lowTag === 'high') {
      return { kind: 'swapped' };
    }
    if (lowTag === 'high') {
      return { kind: 'high-as-low' };
    }
    if (mainTag === 'low') {
      return { kind: 'low-as-main' };
    }

    return null;
  }

  // A single low-tagged expert running the whole schedule is usually the wrong pick.
  return mainTag === 'low' ? { kind: 'single-low' } : null;
};

/** Prefer full H3 Diffusers installs for checkpoint components; a components-only source qualifies when none exists. */
const findH3ComponentSource = (models: readonly ModelConfig[]): MainModelConfig | null => {
  const candidates = models.filter(
    (candidate): candidate is ModelConfig & MainModelConfig =>
      candidate.type === 'main' && candidate.base === 'minimax-h3' && candidate.format === 'diffusers'
  );

  return candidates.find((candidate) => !isComponentsOnlyVideoMain(candidate)) ?? candidates[0] ?? null;
};

const findLtx2ComponentSource = (models: readonly ModelConfig[]): MainModelConfig | null =>
  models.find(
    (candidate): candidate is ModelConfig & MainModelConfig =>
      candidate.type === 'main' && candidate.base === 'ltx-2' && candidate.format === 'diffusers'
  ) ?? null;

/** The Gemma-4 encoder every LTX-2 generation needs; no main model carries one. */
const findLtx2TextEncoder = (models: readonly ModelConfig[]): ModelConfig | null =>
  models.find((candidate) => candidate.type === 'gemma4_encoder' && candidate.base === 'ltx-2') ?? null;

/** Selected so the Auto duration switch is visible; the switch itself stays off. */
const findLtx2DurationHead = (models: readonly ModelConfig[]): ModelConfig | null =>
  models.find((candidate) => candidate.type === 'ltx2_duration_head' && candidate.base === 'ltx-2') ?? null;

export const getDefaultVideoSettings = (
  model?: MainModelConfig,
  models: readonly ModelConfig[] = []
): VideoSettings => {
  const config = getVideoConfig(model);

  const base: VideoSettings = {
    ltx2ExtendContextFrames: LTX2_EXTEND_CONTEXT_FRAMES,
    ltx2DurationHeadModel: model && model.base === 'ltx-2' ? findLtx2DurationHead(models) : null,
    autoDuration: false,
    acceleratorEnabled: false,
    acceleratorLoraKeys: [],
    aspectRatioId: '16:9',
    batchCount: 1,
    cfgScale: config.defaults.cfgScale,
    cfgScaleLowNoise: config.defaults.cfgScaleLowNoise,
    conditioningClip: null,
    ...VIDEO_DYNAMIC_PROMPTS_DEFAULTS,
    audioCfgScale: config.defaults.audioCfgScale,
    // Seed a component source from the catalog so a single-file main starts invokable; reset reuses defaults.
    componentSourceModel:
      model && model.base === 'minimax-h3' && model.format === 'checkpoint'
        ? findH3ComponentSource(models)
        : model && model.base === 'ltx-2' && model.format !== 'diffusers'
          ? findLtx2ComponentSource(models)
          : null,
    firstFrameImage: null,
    fps: config.fps.defaultValue,
    h3TaskOverride: undefined,
    h3HybridBaseModel: null,
    h3HybridStartBlock: MINIMAX_H3_HYBRID_BLOCK_RANGE.defaultStart,
    h3TextEncoderModel: null,
    h3TransformerModel: null,
    lastFrameImage: null,
    loras: [],
    // Required, not an override: seeded so a picked LTX-2 model is invokable.
    ltx2TextEncoderModel: model && model.base === 'ltx-2' ? findLtx2TextEncoder(models) : null,
    modalityScale: config.defaults.modalityScale,
    modelKey: model?.key ?? '',
    negativePrompt: config.defaultNegativePrompt ?? '',
    negativePromptEnabled: true,
    negativePromptHeightPx: 56,
    numFrames: config.frames.defaultValue,
    positivePrompt: '',
    positivePromptHeightPx: 96,
    references: [],
    seed: Math.floor(Math.random() * SEED_MAX),
    seedMode: 'random',
    sourceVideo: null,
    steps: config.defaults.steps,
    stgScale: config.defaults.stgScale,
    targetResolution: config.defaults.targetResolution,
    vae: null,
    wanLowNoiseModel: null,
    wanT5EncoderModel: null,
  };

  // Enable installed accelerators by default, matching bundled templates' practical sampling schedules.
  if (model && config.accelerator) {
    const result = getAcceleratorToggleResult(base, model, models, true);

    if (!result.missingLoras) {
      return result.settings;
    }
  }

  return base;
};

export const getVideoSettingsWithModelDefaults = (
  settings: VideoSettings,
  model: MainModelConfig,
  models: readonly ModelConfig[] = []
): VideoSettings => {
  const modelDefaults = getDefaultVideoSettings(model, models);

  const previousKeys = new Set(settings.acceleratorLoraKeys);

  return {
    ...settings,
    acceleratorEnabled: modelDefaults.acceleratorEnabled,
    acceleratorLoraKeys: modelDefaults.acceleratorLoraKeys,
    aspectRatioId: modelDefaults.aspectRatioId,
    audioCfgScale: modelDefaults.audioCfgScale,
    cfgScale: modelDefaults.cfgScale,
    cfgScaleLowNoise: modelDefaults.cfgScaleLowNoise,
    componentSourceModel: modelDefaults.componentSourceModel,
    fps: modelDefaults.fps,
    h3TaskOverride: undefined,
    h3HybridBaseModel: modelDefaults.h3HybridBaseModel,
    h3HybridStartBlock: modelDefaults.h3HybridStartBlock,
    h3TextEncoderModel: modelDefaults.h3TextEncoderModel,
    h3TransformerModel: modelDefaults.h3TransformerModel,
    loras: [
      ...settings.loras.filter(
        (lora) => !previousKeys.has(lora.model.key) && !modelDefaults.loras.some((d) => d.model.key === lora.model.key)
      ),
      ...modelDefaults.loras,
    ],
    ltx2TextEncoderModel: modelDefaults.ltx2TextEncoderModel,
    modalityScale: modelDefaults.modalityScale,
    modelKey: model.key,
    numFrames: modelDefaults.numFrames,
    references: modelDefaults.references,
    steps: modelDefaults.steps,
    stgScale: modelDefaults.stgScale,
    targetResolution: modelDefaults.targetResolution,
    vae: modelDefaults.vae,
    wanLowNoiseModel: modelDefaults.wanLowNoiseModel,
    wanT5EncoderModel: modelDefaults.wanT5EncoderModel,
  };
};

/** User setup a switch discards for good; switching back restores none of it, so the UI confirms these. */
export type VideoModelSwitchLossKey =
  | 'references'
  | 'initialVideo'
  | 'trimmedInitialVideo'
  | 'conditioningClip'
  | 'firstFrame'
  | 'lastFrame'
  | VideoComponentValueKey;

export interface VideoModelSwitchLoss {
  key: VideoModelSwitchLossKey;
  /** How many items go: references; 1 for a single slot. */
  count: number;
}

/** Reported rather than confirmed: each stays on screen and is one value to re-enter. */
export type VideoModelSwitchAdjustment =
  | 'frames'
  | 'targetResolution'
  | 'fps'
  | 'acceleration'
  | 'cfgLowNoise'
  | 'advancedGuidance'
  | 'steps'
  | 'autoDuration';

export interface VideoModelSelectionResult {
  settings: VideoSettings;
  losses: readonly VideoModelSwitchLoss[];
  adjustments: readonly VideoModelSwitchAdjustment[];
}

/** Previewing and applying a model switch both call this, so they cannot disagree. */
export const getVideoModelSelectionResult = ({
  currentSettings,
  model,
  models,
  h3TaskOverride,
}: {
  currentSettings: VideoSettings;
  model: MainModelConfig;
  models: readonly ModelConfig[];
  /** Task override for the incoming H3 model, applied before the loss preview. */
  h3TaskOverride?: VideoSettings['h3TaskOverride'];
}): VideoModelSelectionResult => {
  const catalogModel = model;
  h3TaskOverride = model.base === 'minimax-h3' && model.format === 'checkpoint' ? h3TaskOverride : undefined;
  model = getEffectiveVideoModel(model, { h3TaskOverride });
  const config = getVideoConfig(model);
  // No modelKey means healing defaults, not user choices: bootstrap the model's defaults first.
  const start = currentSettings.modelKey
    ? currentSettings
    : getVideoSettingsWithModelDefaults(currentSettings, model, models);
  const next: VideoSettings = {
    ...start,
    modelKey: model.key,
    h3TaskOverride,
  };
  const losses: VideoModelSwitchLoss[] = [];
  const adjustments = new Set<VideoModelSwitchAdjustment>();
  const modes = config.modes;
  const previousModel = models.find((entry) => entry.key === currentSettings.modelKey);
  const previousConfig =
    previousModel && isSupportedVideoModel(previousModel)
      ? getVideoConfig(getEffectiveVideoModel(previousModel, currentSettings))
      : null;

  if (next.references.length > 0 && !modes.includes('reference')) {
    // An untouched Initial Video anchor is re-derived on the way back; a set trim or conditioning on it is not.
    const supplied = next.references.filter(
      (entry) =>
        !(
          entry.kind === 'video' &&
          entry.fromSourceVideo &&
          !entry.trimOverridden &&
          entry.conditioning === REFERENCE_EXTEND_ANCHOR_CONDITIONING
        )
    ).length;

    if (supplied > 0) {
      losses.push({ count: supplied, key: 'references' });
    }

    next.references = [];
  }

  if (next.sourceVideo && !modes.includes('extend') && !config.references?.extend) {
    losses.push({ count: 1, key: isVideoSourceClipTrimmed(next.sourceVideo) ? 'trimmedInitialVideo' : 'initialVideo' });
    next.sourceVideo = null;
  }

  if (next.conditioningClip && !modes.includes(resolveVideoMode(next))) {
    // Left behind, this slot would keep driving the canvas and disable the aspect-ratio control.
    next.conditioningClip = null;
    losses.push({ count: 1, key: 'conditioningClip' });
  }

  // Snap frame count before deriving reference context so cross-family switches cannot retain an undersized Wan budget.
  const snappedFrames = snapVideoNumFrames(model, next.numFrames);
  const framesChanged = snappedFrames !== next.numFrames;

  if (framesChanged) {
    next.numFrames = snappedFrames;
    adjustments.add('frames');
  }

  // Derive a missing source anchor on model switch; avoid resetting hand-tuned trims on task-neutral transitions.
  if (
    config.references?.extend &&
    next.sourceVideo &&
    !next.references.some((entry) => entry.kind === 'video' && entry.fromSourceVideo === true)
  ) {
    next.references = applyReferenceExtendSourceVideo(
      next.references,
      next.sourceVideo,
      config.references.maxVideos,
      next.numFrames
    );
  }

  // Rebudget inherited tails only when frame count changes, preserving trims on neutral reselection.
  if (config.references?.extend && framesChanged) {
    next.references = applyReferenceExtendNumFrames(next.references, next.numFrames);
  }

  if (next.firstFrameImage && !modes.includes('first-frame') && !modes.includes('first-last')) {
    next.firstFrameImage = null;
    losses.push({ count: 1, key: 'firstFrame' });
  }

  if (next.lastFrameImage) {
    // The end-frame anchor rides FLF2V with a first frame or source video; alone it needs the last-frame mode.
    const lastFrameSupported =
      next.firstFrameImage || next.sourceVideo ? modes.includes('first-last') : modes.includes('last-frame');

    if (!lastFrameSupported) {
      next.lastFrameImage = null;
      losses.push({ count: 1, key: 'lastFrame' });
    }
  }

  if (!config.targetResolutions.some((option) => option.id === next.targetResolution)) {
    next.targetResolution = config.defaults.targetResolution;
    adjustments.add('targetResolution');
  }

  const clampedFps =
    config.fps.editable && Number.isFinite(next.fps)
      ? Math.min(config.fps.max, Math.max(config.fps.min, Math.round(next.fps)))
      : config.fps.defaultValue;

  if (clampedFps !== next.fps) {
    next.fps = clampedFps;
    adjustments.add('fps');
  }

  if (next.acceleratorEnabled) {
    // Keep tuned acceleration while its LoRAs stay compatible; otherwise reapply or restore normal sampling.
    if (!isRecordedAcceleratorIntact(next, model, models, config)) {
      const targetEntries = config.accelerator ? findAcceleratorLoraEntries(model, models) : null;
      const result = getAcceleratorToggleResult(next, model, models, targetEntries !== null);

      Object.assign(next, result.settings);
      adjustments.add('acceleration');
    }
  }

  if (next.cfgScaleLowNoise !== null && !config.cfg.lowNoiseVisible) {
    next.cfgScaleLowNoise = null;
    adjustments.add('cfgLowNoise');
  }

  // Like CFG (Low Noise): a hidden control carries no value; a shown one starts from its family's default.
  const guidanceTransitions = [
    ['audioCfgScale', config.guidance.audioVisible],
    ['stgScale', config.guidance.stgVisible],
    ['modalityScale', config.guidance.modalityVisible],
  ] as const;

  for (const [key, visible] of guidanceTransitions) {
    const wanted = visible ? (next[key] ?? config.defaults[key]) : null;

    if (wanted !== next[key]) {
      const dropped = next[key] !== null;

      next[key] = wanted;
      // Only a value the panel takes away is a clearing; all three share the "Advanced guidance" label.
      if (dropped) {
        adjustments.add('advancedGuidance');
      }
    }
  }

  // Seed the family's negative list into a blank field; dev would otherwise guide against "" at CFG 3. A blank
  // on a variant that shows the field is the user's, unless a detour through one that hides it lost the difference.
  if (config.defaultNegativePrompt && next.negativePromptEnabled && !next.negativePrompt.trim()) {
    const previousCarriesNone = previousModel ? previousConfig?.negativePrompt.usage === 'never' : true;

    if (previousCarriesNone) {
      next.negativePrompt = config.defaultNegativePrompt;
    }
  }

  // A fixed-schedule checkpoint's step count is not the user's to carry over.
  if (!config.stepsEditable && next.steps !== config.defaults.steps) {
    next.steps = config.defaults.steps;
    adjustments.add('steps');
  }

  // A control the previous variant hid, still at that variant's static recipe, was not chosen: re-default it.
  // Compare to the static recipe, not getDefaultVideoSettings (it applies accelerator numbers). Known gap: a value
  // deliberately set to the hiding variant's default is replaced.
  if (!next.acceleratorEnabled && previousConfig) {
    if (!previousConfig.stepsEditable && next.steps === previousConfig.defaults.steps) {
      next.steps = config.defaults.steps;
    }

    if (!previousConfig.cfg.visible && next.cfgScale === previousConfig.defaults.cfgScale) {
      next.cfgScale = config.defaults.cfgScale;
    }
  }

  // Concepts stay, as in Generate: the graph loads only compatible ones (`getActiveCompatibleLoras`).

  const policy = getVideoComponentSectionPolicy(catalogModel, next);
  const slotsByKey = new Map(policy.slots.map((slotPolicy) => [slotPolicy.key, slotPolicy]));
  // A component the previous model seeds again on the way back is not a loss.
  let previousSeeds: VideoSettings | null | undefined;

  for (const key of Object.keys(VIDEO_COMPONENT_SETTING_LABELS) as VideoComponentValueKey[]) {
    const value = next[key];

    if (!value) {
      continue;
    }

    const slotPolicy = slotsByKey.get(key);
    const isCompatible =
      slotPolicy &&
      (!slotPolicy.filter || slotPolicy.filter(value as ModelConfig, getVideoComponentPolicyContext(model, next)));

    if (!isCompatible) {
      next[key] = null;
      previousSeeds ??= previousConfig ? getDefaultVideoSettings(previousModel as MainModelConfig, models) : null;

      if (previousSeeds?.[key]?.key !== value.key) {
        losses.push({ count: 1, key });
      }
    }
  }

  // Auto duration belongs to its head; a head filled in later must not turn it back on.
  if (!next.ltx2DurationHeadModel) {
    if (next.autoDuration) {
      adjustments.add('autoDuration');
    }

    next.autoDuration = false;
  }

  // Autofill required H3 components only when empty; preserve explicit compatible choices.
  if (model.base === 'minimax-h3' && model.format === 'checkpoint' && !next.componentSourceModel) {
    next.componentSourceModel = findH3ComponentSource(models);
  }

  if (model.base === 'ltx-2') {
    if (model.format !== 'diffusers' && !next.componentSourceModel) {
      next.componentSourceModel = findLtx2ComponentSource(models);
    }
    if (!next.ltx2TextEncoderModel) {
      next.ltx2TextEncoderModel = findLtx2TextEncoder(models);
    }
    if (!next.ltx2DurationHeadModel) {
      next.ltx2DurationHeadModel = findLtx2DurationHead(models);
    }
  }

  return { adjustments: [...adjustments], losses, settings: next };
};

const VIDEO_MODE_DESCRIPTIONS: Record<VideoGenerationMode, string> = {
  'audio-to-video': 'generating video for an existing soundtrack',
  extend: 'extending a video',
  'first-frame': 'starting from a first frame',
  'first-last': 'first-to-last-frame interpolation',
  'last-frame': 'ending on a last frame',
  reference: 'reference-conditioned generation',
  txt2vid: 'text-to-video',
  'video-to-audio': 'generating a soundtrack for an existing clip',
};

const hasModelKey = (models: readonly ModelConfig[], key: string, type?: string): boolean =>
  models.some((model) => model.key === key && (!type || model.type === type));

export type VideoSection =
  | 'components'
  | 'concepts'
  | 'conditioningClip'
  | 'dimensions'
  | 'frames'
  | 'guidance'
  | 'references'
  | 'render'
  | 'source';

export type VideoField =
  | 'accelerator'
  | 'audioCfg'
  | 'cfg'
  | 'cfgLowNoise'
  | 'components'
  | 'concepts'
  | 'conditioningClip'
  | 'extendContext'
  | 'firstFrame'
  | 'fps'
  | 'initialVideo'
  | 'lastFrame'
  | 'modality'
  | 'numFrames'
  | 'references'
  | 'steps'
  | 'stg'
  | 'targetResolution';

const VIDEO_FIELD_SECTIONS: Record<VideoField, VideoSection> = {
  accelerator: 'render',
  audioCfg: 'guidance',
  cfg: 'render',
  cfgLowNoise: 'render',
  components: 'components',
  concepts: 'concepts',
  conditioningClip: 'conditioningClip',
  extendContext: 'source',
  firstFrame: 'frames',
  fps: 'dimensions',
  initialVideo: 'source',
  lastFrame: 'frames',
  modality: 'guidance',
  numFrames: 'dimensions',
  references: 'references',
  steps: 'render',
  stg: 'guidance',
  targetResolution: 'dimensions',
};

/** A conflict between two media lists both fields; none when only the model picker can fix it. */
export interface VideoValidationIssue {
  fields: readonly VideoField[];
  message: string;
}

const when = (condition: unknown, field: VideoField): VideoField[] => (condition ? [field] : []);

/** Unsupported modes are fixed by clearing their media; text-to-video by adding media. */
const VIDEO_MODE_FIELDS: Record<VideoGenerationMode, readonly VideoField[]> = {
  'audio-to-video': ['conditioningClip'],
  extend: ['initialVideo'],
  'first-frame': ['firstFrame'],
  'first-last': ['firstFrame', 'lastFrame'],
  'last-frame': ['lastFrame'],
  reference: ['references'],
  txt2vid: ['conditioningClip', 'firstFrame', 'initialVideo', 'lastFrame', 'references'],
  'video-to-audio': ['conditioningClip'],
};

/** Shown, editable controls; issues count only in these sections, so a count never points at a hidden control. */
export const getVideoPanelFields = (
  model: MainModelConfig | undefined,
  settings: VideoSettings,
  policy: VideoModelPolicy = getVideoModelPolicy(model, settings)
): ReadonlySet<VideoField> => {
  const { modes, ui } = policy;
  const supportsExtend = modes.includes('extend');
  const fields: [VideoField, boolean][] = [
    ['accelerator', Boolean(model && ui.accelerator)],
    ['audioCfg', ui.audioCfgVisible],
    ['cfg', ui.cfgVisible],
    ['cfgLowNoise', ui.cfgLowNoiseVisible],
    ['components', Boolean(model) && getVideoComponentSectionPolicy(model, settings).slots.length > 0],
    ['concepts', true],
    ['conditioningClip', modes.includes('audio-to-video') || modes.includes('video-to-audio')],
    ['extendContext', ui.extendContext !== null],
    ['firstFrame', modes.includes('first-frame') || modes.includes('first-last')],
    // Locked to the initial video while extending, and to a clip that sets the rate.
    [
      'fps',
      ui.fpsVisible && resolveVideoMode(settings) !== 'extend' && !getEffectiveVideoTiming(model, settings).fpsFromClip,
    ],
    ['initialVideo', supportsExtend || Boolean(policy.references?.extend)],
    ['lastFrame', modes.includes('first-last') || modes.includes('last-frame')],
    ['modality', ui.modalityVisible],
    ['numFrames', true],
    ['references', modes.includes('reference')],
    ['steps', ui.stepsEditable],
    ['stg', ui.stgVisible],
    ['targetResolution', true],
  ];

  return new Set(fields.flatMap(([field, shown]) => (shown ? [field] : [])));
};

export const getVideoIssueSections = (issue: VideoValidationIssue, shown: ReadonlySet<VideoField>): VideoSection[] => [
  ...new Set(issue.fields.flatMap((field) => (shown.has(field) ? [VIDEO_FIELD_SECTIONS[field]] : []))),
];

export const getVideoModelAvailabilityIssues = (
  model: MainModelConfig,
  settings: VideoSettings,
  models: readonly ModelConfig[]
): VideoValidationIssue[] => {
  const issues: VideoValidationIssue[] = [];

  if (!hasModelKey(models, model.key, model.type)) {
    issues.push({ fields: [], message: `Selected model "${model.name}" is no longer installed.` });
  }

  for (const key of Object.keys(VIDEO_COMPONENT_SETTING_LABELS) as VideoComponentValueKey[]) {
    const value = settings[key];

    if (value && !hasModelKey(models, value.key, value.type)) {
      issues.push({
        fields: ['components'],
        message: `${VIDEO_COMPONENT_SETTING_LABELS[key]} "${value.name}" is no longer installed.`,
      });
    }
  }

  // Only LoRAs the graph would load need to be installed; disabled or incompatible ones are skipped, as in Generate.
  for (const lora of settings.loras) {
    if (
      lora.isEnabled &&
      isLoraCompatibleWithModel(lora.model, model) &&
      !hasModelKey(models, lora.model.key, 'lora')
    ) {
      // The panel's synced list has already dropped the row, so no section can show it.
      issues.push({ fields: [], message: `LoRA "${lora.model.name}" is no longer installed.` });
    }
  }

  return issues;
};

/** Why Video cannot run with this model, each with the controls that fix it, in the order Invoke lists them. */
export const getVideoValidationIssues = (model: MainModelConfig, settings: VideoSettings): VideoValidationIssue[] => {
  const catalogModel = model;
  model = getEffectiveVideoModel(model, settings);
  // The model picker fixes these, and it sits in no section.
  if (!isSupportedVideoModel(model)) {
    return [{ message: 'Video needs a supported video model before it can be invoked.', fields: [] }];
  }

  if (isComponentsOnlyVideoMain(model)) {
    const family = model.base === 'ltx-2' ? 'LTX-2' : 'MiniMax H3';

    return [
      {
        message: `${model.name} is a components-only install. Select a single-file ${family} transformer as the model; this install then provides its components.`,
        fields: [],
      },
    ];
  }
  if (model.base === 'minimax-h3' && model.format === 'diffusers' && model.variant === 'ref2va') {
    return [
      {
        message: `${model.name} is a Ref2VA folder install, whose transformer weights cannot be folder-loaded. Select a single-file Ref2VA transformer as the model; this install can serve as its Model Components.`,
        fields: [],
      },
    ];
  }

  const config = getVideoConfig(model);
  const issues: VideoValidationIssue[] = [];
  const mode = resolveVideoMode(settings);
  const referenceOnly = config.modes.length === 1 && config.modes[0] === 'reference';

  if (settings.firstFrameImage && settings.sourceVideo) {
    issues.push({
      fields: ['firstFrame', 'initialVideo'],
      message: 'A first frame and an initial video cannot be combined. Clear one of them.',
    });
  }

  if (settings.conditioningClip && (settings.sourceVideo || settings.references.length > 0)) {
    // `resolveVideoMode` would silently drop one rather than run something it cannot express.
    issues.push({
      fields: [
        'conditioningClip',
        ...when(settings.sourceVideo, 'initialVideo'),
        ...when(settings.references.length > 0, 'references'),
      ],
      message: 'A conditioning clip cannot be combined with an initial video or references. Clear one side.',
    });
  }

  if (
    isConditioningClipExcludingFrames(settings.conditioningClip) &&
    (settings.firstFrameImage || settings.lastFrameImage)
  ) {
    // The clip's picture fills every frame; its soundtrack does not, so frames stay available in that role.
    issues.push({
      fields: [
        'conditioningClip',
        ...when(settings.firstFrameImage, 'firstFrame'),
        ...when(settings.lastFrameImage, 'lastFrame'),
      ],
      message: "A conditioning clip's picture cannot be combined with first/last frames. Clear one side.",
    });
  }

  if (settings.references.length > 0 && (settings.firstFrameImage || settings.lastFrameImage)) {
    issues.push({
      fields: [
        'references',
        ...when(settings.firstFrameImage, 'firstFrame'),
        ...when(settings.lastFrameImage, 'lastFrame'),
      ],
      message: 'References cannot be combined with first/last frames. Clear one side.',
    });
  }

  if (settings.references.length > 0 && settings.sourceVideo && !config.references?.extend) {
    issues.push({
      fields: ['references', 'initialVideo'],
      message: 'References cannot be combined with an initial video on this model. Clear one side.',
    });
  }

  if (!config.modes.includes(mode)) {
    if (referenceOnly && settings.references.length === 0) {
      issues.push({ fields: ['references'], message: 'Reference-to-video needs at least one reference.' });
    } else {
      issues.push({
        fields: VIDEO_MODE_FIELDS[mode],
        message: `${model.name} does not support ${VIDEO_MODE_DESCRIPTIONS[mode]}.`,
      });
    }
  } else if (mode === 'extend' && settings.lastFrameImage && !config.modes.includes('first-last')) {
    issues.push({
      fields: ['initialVideo', 'lastFrame'],
      message: `${model.name} cannot target a destination image while extending a video.`,
    });
  }

  if (mode === 'reference') {
    const caps = config.references;
    const videoCount = settings.references.filter((reference) => reference.kind === 'video').length;
    const imageCount = settings.references.length - videoCount;
    if (caps && videoCount > caps.maxVideos) {
      issues.push({ fields: ['references'], message: `At most ${caps.maxVideos} video references are supported.` });
    }
    if (caps && imageCount > caps.maxImages) {
      issues.push({ fields: ['references'], message: `At most ${caps.maxImages} image references are supported.` });
    }
    for (const reference of settings.references) {
      if (reference.kind !== 'video') {
        continue;
      }
      if (!Number.isInteger(reference.clip.startFrame) || !Number.isInteger(reference.clip.endFrame)) {
        issues.push({ fields: ['references'], message: 'Reference video trim bounds must be whole frame numbers.' });
        break;
      }
      if (
        reference.clip.startFrame < 0 ||
        reference.clip.endFrame > reference.clip.numFrames - 1 ||
        reference.clip.endFrame < reference.clip.startFrame
      ) {
        issues.push({ fields: ['references'], message: 'A reference video trim is outside its clip.' });
        break;
      }
    }
  }

  // Effective values: a conditioning clip overrides the stored ones, which stay the user's.
  const timing = getEffectiveVideoTiming(model, settings);

  if (!isValidVideoNumFrames(model, timing.numFrames)) {
    const framesFloor = config.frames.kind === 'grid' ? config.frames.min : (config.frames.choices[0] ?? 0);
    const framesCeiling =
      config.frames.kind === 'grid'
        ? config.frames.max
        : (config.frames.choices[config.frames.choices.length - 1] ?? 0);

    issues.push({
      fields: timing.numFramesFromClip
        ? ['conditioningClip', ...when(settings.conditioningClip?.role === 'audio', 'fps')]
        : ['numFrames'],
      message: timing.numFramesFromClip
        ? `The conditioning clip works out to ${timing.numFrames} frames, outside the ${framesFloor}-${framesCeiling} ${model.name} generates. Use a ${timing.numFrames < framesFloor ? 'longer' : 'shorter'} clip${settings.conditioningClip?.role === 'audio' ? ', or change the frame rate' : ''}.`
        : config.frames.kind === 'grid'
          ? `Frame count must be between ${config.frames.min} and ${config.frames.max} in steps of ${config.frames.step} (${describeVideoFramesGrid(config.frames)}).`
          : `Frame count must be one of the ${model.name} grid values (17·n + 5, ${config.frames.choices[0]}–${config.frames.choices[config.frames.choices.length - 1]}).`,
    });
  }

  // Backend fps/steps are int fields (pydantic fails at enqueue); a clip's rate is exempt, LTX-2 fps is a float.
  if (
    !Number.isFinite(timing.fps) ||
    (!timing.fpsFromClip && !Number.isInteger(timing.fps)) ||
    timing.fps < config.fps.min ||
    timing.fps > config.fps.max
  ) {
    issues.push({
      fields: timing.fpsFromClip ? ['conditioningClip'] : ['fps'],
      message: timing.fpsFromClip
        ? `The conditioning clip runs at ${timing.fps} fps, outside the ${config.fps.min}-${config.fps.max} fps range ${model.name} supports.`
        : config.fps.editable
          ? `FPS must be a whole number between ${config.fps.min} and ${config.fps.max}.`
          : `${model.name} generates at a fixed ${config.fps.defaultValue} FPS.`,
    });
  }

  if (!Number.isInteger(settings.steps) || settings.steps < config.minSteps) {
    issues.push({ fields: ['steps'], message: `Steps must be a whole number of at least ${config.minSteps}.` });
  }

  const preset = config.targetResolutions.find((option) => option.id === settings.targetResolution);

  // See `isTwoStageSupportedForMode`. Caught here, before the base pass and the upscale have been paid for.
  if (preset?.stages === 2 && !isTwoStageSupportedForMode(mode)) {
    issues.push({
      fields: ['targetResolution', 'conditioningClip'],
      message: `${VIDEO_MODE_DESCRIPTIONS[mode][0]?.toUpperCase()}${VIDEO_MODE_DESCRIPTIONS[mode].slice(1)} does not run a two-stage target resolution. Pick a single-stage resolution, or clear the conditioning clip.`,
    });
  }

  // A one-step refine can only enter at the schedule's bottom, which the backend refuses after the base pass.
  if (preset?.stages === 2 && Number.isInteger(settings.steps) && settings.steps < LTX2_MIN_TWO_STAGE_STEPS) {
    issues.push({
      fields: ['targetResolution', 'steps'],
      message:
        `A two-stage target resolution needs at least ${LTX2_MIN_TWO_STAGE_STEPS} steps; the second pass ` +
        `refines what the first produced and cannot run in one.`,
    });
  }

  // The crossfade consumes context frames from both halves, so both sides have floors that otherwise fail late.
  if (mode === 'extend' && model.base === 'ltx-2' && settings.sourceVideo) {
    const context = settings.ltx2ExtendContextFrames;

    if (settings.numFrames <= context) {
      issues.push({
        fields: ['numFrames', 'extendContext'],
        message:
          `A continuation opens with ${context} frames of the source, so anything at or ` +
          `below that would be all replay and no continuation. Raise Frames above ${context}.`,
      });
    }

    const affordable = ltx2MaxExtendContextFrames(settings.sourceVideo.width, settings.sourceVideo.height);

    if (affordable === 0) {
      issues.push({
        fields: ['initialVideo'],
        message:
          `The join blends the held frames of the initial video at its own ` +
          `${settings.sourceVideo.width}x${settings.sourceVideo.height}, which needs more memory than it is ` +
          `allowed even at the smallest context. Use a source at or below about 2560x1440.`,
      });
    } else if (context > affordable) {
      // Reachable via recalled or stored settings against a larger source; the control itself is bounded.
      issues.push({
        fields: ['extendContext'],
        message:
          `Context Frames is ${context}, but blending that many frames of a ` +
          `${settings.sourceVideo.width}x${settings.sourceVideo.height} source needs more memory than the join ` +
          `is allowed. Lower it to ${affordable} or less.`,
      });
    }

    const kept = settings.sourceVideo.endFrame - settings.sourceVideo.startFrame + 1;

    if (kept < context) {
      issues.push({
        fields: ['initialVideo', 'extendContext'],
        message:
          `The join blends ${context} frames out of each half, but the initial video's trim ` +
          `keeps only ${kept}. Keep at least ${context} frames of it, or lower Context Frames.`,
      });
    }
  }

  if (config.cfg.visible && (!Number.isFinite(settings.cfgScale) || settings.cfgScale < 1)) {
    issues.push({ fields: ['cfg'], message: 'CFG must be at least 1.' });
  }

  if (config.cfg.lowNoiseVisible && settings.cfgScaleLowNoise !== null && settings.cfgScaleLowNoise < 1) {
    issues.push({ fields: ['cfgLowNoise'], message: 'CFG (Low Noise) must be at least 1.' });
  }

  // The denoise node's own `ge` floors; below them pydantic fails at enqueue.
  const guidanceBounds = [
    {
      field: 'audioCfg' as const,
      floor: 1,
      label: 'Audio CFG',
      value: config.guidance.audioVisible ? settings.audioCfgScale : null,
    },
    { field: 'stg' as const, floor: 0, label: 'STG', value: config.guidance.stgVisible ? settings.stgScale : null },
    {
      field: 'modality' as const,
      floor: 1,
      label: 'Modality guidance',
      value: config.guidance.modalityVisible ? settings.modalityScale : null,
    },
  ];

  for (const { field, floor, label, value } of guidanceBounds) {
    if (value !== null && (!Number.isFinite(value) || value < floor)) {
      issues.push({ fields: [field], message: `${label} must be at least ${floor}.` });
    }
  }

  if (settings.acceleratorEnabled && !config.accelerator) {
    issues.push({
      fields: ['accelerator'],
      message: `${model.name} has no distillation fast path. Turn the accelerator off to generate with it.`,
    });
  }

  // extract_video_range bounds are int fields; a fractional persisted value fails pydantic at enqueue.
  if (
    settings.sourceVideo &&
    (!Number.isInteger(settings.sourceVideo.startFrame) || !Number.isInteger(settings.sourceVideo.endFrame))
  ) {
    issues.push({ fields: ['initialVideo'], message: 'The initial video trim bounds must be whole frame numbers.' });
  } else if (settings.sourceVideo) {
    const { endFrame, numFrames, startFrame } = settings.sourceVideo;

    // Require valid source bounds and at least two frames for crossfade before encoding.
    if (startFrame < 0 || endFrame > numFrames - 1 || endFrame - startFrame + 1 < MIN_VIDEO_TRIM_FRAMES) {
      issues.push({
        fields: ['initialVideo'],
        message: 'The initial video trim must keep at least two frames within the clip.',
      });
    }

    if (numFrames < MIN_VIDEO_TRIM_FRAMES) {
      issues.push({ fields: ['initialVideo'], message: 'The initial video is too short to extend.' });
    }

    // Wan extension inherits source fps; validate the backend's 1–120 range before expensive denoising.
    if (model.base === 'wan') {
      const inheritedFps = Math.round(settings.sourceVideo.fps);

      if (inheritedFps < 1 || inheritedFps > 120) {
        issues.push({
          fields: ['initialVideo'],
          message: `The initial video's frame rate (${settings.sourceVideo.fps} fps) is outside the 1-120 fps range Wan extension supports.`,
        });
      }
    }
  }

  if (!getVideoDimensions(model, settings)) {
    issues.push({
      fields: [
        ...when(settings.firstFrameImage, 'firstFrame'),
        ...when(settings.lastFrameImage, 'lastFrame'),
        ...when(settings.references.length > 0, 'references'),
        ...when(settings.sourceVideo, 'initialVideo'),
        ...when(settings.conditioningClip, 'conditioningClip'),
      ],
      message:
        model.base === 'minimax-h3'
          ? 'MiniMax H3 supports aspect ratios from 1:4 to 4:1. The conditioning media is outside that range.'
          : 'The conditioning media is too small or degenerate to derive video dimensions from.',
    });
  }

  const componentPolicy = getVideoComponentSectionPolicy(catalogModel, settings);
  issues.push(
    ...componentPolicy
      .validate(getVideoComponentPolicyContext(model, settings))
      .map((message): VideoValidationIssue => ({ fields: ['components'], message }))
  );

  return issues;
};

export const getVideoValidationReasons = (model: MainModelConfig, settings: VideoSettings): string[] =>
  getVideoValidationIssues(model, settings).map((issue) => issue.message);
