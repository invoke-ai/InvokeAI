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
} from './settings';

// Wan variants differ in conditioning, pixel grid, and experts; key capabilities by base and variant.

export type SupportedVideoBase = 'wan' | 'minimax-h3' | 'ltx-2';

export type VideoNegativePromptUsage = 'always' | 'cfg-gated' | 'never';

export interface VideoTargetResolutionOption {
  id: VideoTargetResolution;
  label: string;
  /**
   * How many denoise passes the preset runs. Absent means one. A two-stage preset generates at half
   * this canvas, upscales the latent x2 and refines it, which is what buys detail no single pass at
   * this size would reach -- and it costs two passes, the second at four times the token count.
   */
  stages?: 2;
}

/**
 * Extends the shape `snapNumFramesToGrid` takes, so the two cannot drift: the snapper is the
 * production consumer of this policy, and a field added here has to remain assignable to it.
 */
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

/**
 * Family accelerator LoRAs and sampling defaults. steps describes the reference release; getAcceleratorSteps
 * resolves release-specific schedules.
 */
export interface VideoAcceleratorConfig {
  label: 'Lightning' | 'Turbo' | 'Distilled';
  steps: number;
  /**
   * Step counts for releases whose names carry no "N-step" token. First match
   * wins; consulted only when the name does not state the count itself.
   */
  stepOverrides?: readonly { pattern: RegExp; steps: number }[];
  cfgScale: number;
  cfgScaleLowNoise: number | null;
  /**
   * Guidance values the accelerator requires, for families that expose more than the primary CFG
   * scale. A step-distilled model is trained to predict without guidance at all, so leaving the
   * extra scales at their guided defaults does not merely waste compute — it pushes the sample off
   * the distribution the distillation was fitted to. Absent for accelerators whose family has no
   * extra guidance, in which case the toggle leaves those settings alone.
   */
  guidance?: { audioCfgScale: number; modalityScale: number; stgScale: number };
}

interface VideoVariantConfig {
  modes: readonly VideoGenerationMode[];
  pixelMultiple: number;
  targetResolutions: readonly VideoTargetResolutionOption[];
  /**
   * The step budget the refine pass of a two-stage preset runs at, when this variant needs one of
   * its own. The schedule is entered partway down and truncated, so a budget here buys fewer refine
   * steps than the same number would in a base pass: measured on a W7900, a guided dev step at the
   * 1024p canvas is 262 s, so inheriting `defaults.steps` (30 -> 17 refine steps) would be 74
   * minutes there and over eight hours at 1536p. Absent leaves the base pass's budget alone, which
   * is right for the distilled checkpoint: its schedule is fixed and a budget is ignored.
   */
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
  /** False when the checkpoint's schedule is fixed and the step count is not the user's to set. */
  stepsEditable: boolean;
  frames: VideoFramesPolicy;
  fps: VideoFpsPolicy;
  cfg: { visible: boolean; lowNoiseVisible: boolean };
  /** The guidance terms beyond the primary CFG scale that this family exposes. */
  guidance: { audioVisible: boolean; stgVisible: boolean; modalityVisible: boolean };
  /** What a fresh panel seeds the negative prompt with, when the family ships one. */
  defaultNegativePrompt?: string;
  negativePrompt: { visible: boolean; usage: VideoNegativePromptUsage };
  /** The family's released prompt enhancer, which Expand Prompt preselects. */
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

/** A family whose only guidance control is the primary CFG scale. */
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

/**
 * The LTX-2.5 step-distillation LoRA: 8 guidance-free steps against Dev's ~30 guided ones, so every guidance
 * scale goes to its identity. `steps: 8` is stated because the release name only carries the rank (450).
 */
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
  // The T2V expert pair has no reference-image conditioning channels.
  t2v_a14b: { ...WAN_A14B_COMMON, modes: ['txt2vid'] },
  // I2V experts require a reference frame; extension uses the source's last frame through FLF2V conditioning.
  i2v_a14b: { ...WAN_A14B_COMMON, modes: ['first-frame', 'first-last', 'extend'] },
  // TI2V-5B has text/image conditioning, no end-frame channel, and one expert.
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

// An unknown Wan variant (new backend release) gets the most permissive A14B
// capabilities rather than being blocked: the backend probe is the authority.
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

// Upstream Ref2VA declares only reference-conditioned generation.
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
  // Every conditioning shape the family supports: text-to-video, a held first and/or last frame,
  // continuing an existing clip, and the two whole-modality modes -- all with a generated
  // soundtrack.
  modes: ['txt2vid', 'first-frame', 'last-frame', 'first-last', 'extend', 'audio-to-video', 'video-to-audio'] as const,
  pixelMultiple: LTX2_CANVAS_MULTIPLE,
  targetResolutions: LTX2_TARGET_RESOLUTION_OPTIONS,
};

/**
 * The dev checkpoint: a guided 30-step schedule with all four of LTX-2's
 * guidance terms. The scales are the released pipeline's for this generation
 * (ltx-pipelines `PipelineParams`), including an audio CFG far above the
 * video's — audio follows the prompt much less readily.
 */
const LTX2_DEV: VideoVariantConfig = {
  ...LTX2_COMMON,
  // Only Dev offers it. The distilled *checkpoint* already is the fast path, and stacking the LoRA
  // on top of it would patch a model the distillation was not fitted to.
  accelerator: LTX2_DISTILLED_ACCELERATOR,
  // The release guides against this list, not against an empty string: it is as much a part of the
  // recipe as the scales are. Mirrors LTX2_DEFAULT_NEGATIVE_PROMPT in
  // invokeai/backend/ltx2/constants.py, which is the node's own default.
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
  // Dev pays four forwards a step -- cond, uncond, STG and modality -- so the refine pass gets a
  // budget of its own rather than the 30 the base pass runs. This is the count the pass actually
  // samples, not a schedule resolution it is truncated out of.
  refineSteps: 8,
  stepsEditable: true,
};

/**
 * The guidance-distilled checkpoint: eight fixed noise levels, one forward per
 * step, and no guidance at all — the denoise node ignores the scales rather
 * than letting them wreck the output, so the panel offers none of them and
 * shows the step count as the fixed value it is.
 */
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
  // Register Ref2VA explicitly so it never inherits FL2VA fallback modes.
  // Dev is the fallback: an unrecognised variant gets the guided schedule, which is merely slow on a distilled
  // checkpoint, where the reverse produces noise.
  'ltx-2': { fallback: LTX2_DEV, variants: { ltx2_dev: LTX2_DEV, ltx2_distilled: LTX2_DISTILLED } },
  'minimax-h3': { fallback: MINIMAX_H3_FL2VA, variants: { fl2va: MINIMAX_H3_FL2VA, ref2va: MINIMAX_H3_REF2VA } },
  wan: { fallback: WAN_FALLBACK_VARIANT, variants: WAN_VARIANTS },
};

export const SUPPORTED_VIDEO_BASES = Object.keys(VIDEO_GENERATION) as SupportedVideoBase[];

/**
 * Supports Wan mains and H3 folder/checkpoint mains. H3 checkpoint variant owns task identity; a Diffusers source
 * supplies its remaining components.
 */
export const isSupportedVideoModel = <T extends { base: string; type: string; format?: string }>(
  model: T
): model is T & MainModelConfig =>
  model.type === 'main' &&
  (model.base === 'wan' ||
    ((model.base === 'minimax-h3' || model.base === 'ltx-2') &&
      (model.format === 'diffusers' || model.format === 'checkpoint')));

/**
 * Select runnable identity-bearing models. Components-only and Ref2VA Diffusers folders remain supported for
 * stored state but serve only as component sources, not selectable mains.
 */
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

// Fallback keeps UI selectors crash-safe while nothing is selected;
// isSupportedVideoModel() still blocks invocation.
const FALLBACK_VARIANT_CONFIG = WAN_FALLBACK_VARIANT;

const getVideoConfig = (
  model: Pick<MainModelConfig, 'base' | 'type' | 'variant' | 'format'> | undefined
): VideoVariantConfig => getVideoVariantConfig(model) ?? FALLBACK_VARIANT_CONFIG;

/**
 * Whether a mode can run a two-stage target resolution. A conditioning clip cannot: `ltx2_denoise`
 * refuses a refine pass alongside a held-clean modality, because the refine pass re-noises every
 * token and the held stream would have to be re-encoded at the refine canvas the way a first frame
 * is. Asked by validation and by the graph-coverage matrix, so the two cannot disagree.
 */
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

/**
 * The preset a model will actually run, which is not always the one the settings hold: a record
 * persisted under another family keeps its own preset (`normalizeVideoSettings` accepts any
 * family's), and only a model *selection* re-coerces it. Every consumer has to agree on this one
 * answer -- `getVideoDimensions` resolves its canvas from it, so a caller reading the raw value
 * would be describing a different run than the dimensions it was handed.
 */
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

/**
 * Conditioning media overrides preset aspect ratio; target resolution still applies. Returns null for unsupported
 * ratios or degenerate inputs.
 */
export const getVideoDimensions = (
  model: MainModelConfig | undefined,
  settings: Pick<
    VideoSettings,
    'aspectRatioId' | 'targetResolution' | 'firstFrameImage' | 'lastFrameImage' | 'sourceVideo' | 'conditioningClip'
  >
): ResolvedVideoDimensions | null => {
  const config = getVideoConfig(model);
  const targetResolution = coerceTargetResolution(config, settings.targetResolution);

  // Only in the `video` role: there the clip's picture *is* the generation, fitted to the canvas
  // the preset picks for its ratio. In the `audio` role the picture is what gets made, so the
  // aspect-ratio control still owns the frame.
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
      ? resolveMiniMaxH3Canvas(inputs.width, inputs.height, targetResolution as '768 highres' | '768 lowres')
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

/**
 * The frame count and frame rate a run will actually use. A conditioning clip decides both for
 * the modality it holds clean: its picture is as long as it is, and its soundtrack covers as many
 * frames as the chosen rate spans. The panel's own stored values are left untouched underneath, so
 * clearing the clip restores what the user had set.
 *
 * Only ever the clip's when the model supports the mode -- a clip left behind by a model switch is
 * stale state the panel offers to clear, not a timing source.
 */
export interface EffectiveVideoTiming {
  numFrames: number;
  fps: number;
  /** True when the conditioning clip decided the value, i.e. the control is showing a derived number. */
  numFramesFromClip: boolean;
  fpsFromClip: boolean;
}

/**
 * Whether this mode has a length left for the duration head to decide.
 *
 * A conditioning clip drives `num_frames` by its own edge, and a second edge into one input is a
 * malformed graph. An extension has no such edge, but its frame count is the new material the user
 * asked for, measured against the source it joins -- a prompt's natural length is not that number.
 *
 * Separate from `isAutoDurationActive` because the panel needs to say *why* the control is
 * unavailable, which "not active" alone cannot tell it.
 */
export const isAutoDurationSupportedForMode = (
  settings: Pick<
    VideoSettings,
    'conditioningClip' | 'firstFrameImage' | 'lastFrameImage' | 'references' | 'sourceVideo'
  >
): boolean => {
  const mode = resolveVideoMode(settings);

  return mode !== 'audio-to-video' && mode !== 'video-to-audio' && mode !== 'extend';
};

/**
 * Whether the duration head will choose this run's length: the user asked for it, a head is
 * selected, and the mode has a length left to decide.
 */
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

  // The picture is the given one, so its own rate is the run's -- but only when the gallery knew
  // that rate. A generated picture is made at whatever rate the panel asks for, and only the
  // length follows from the soundtrack.
  const fpsFromClip = conditioning.role === 'video' && conditioning.fpsKnown;
  const fps = fpsFromClip ? conditioning.clip.fps : settings.fps;

  return {
    fps,
    fpsFromClip,
    numFrames: ltx2FramesForClip(conditioning, fps),
    numFramesFromClip: true,
  };
};

/**
 * Whether the run is guidance-free because an accelerator made it so.
 *
 * An accelerator that declares a guidance triple drives every scale to its identity AND makes the
 * graph send `schedule: 'distilled'`. The backend then *discards* the guidance scales and the step
 * count outright (`_resolve_guidance` and the step clamp in `ltx2_denoise.py`, both of which log
 * that they are ignoring what was asked). So those controls are not merely redundant while the
 * accelerator is on -- they cannot take effect at all, and the panel must stop offering them.
 *
 * Which is to say: an accelerated Dev model *is* the family's distilled variant, and presents like
 * it. Families whose accelerator declares no guidance (Wan, MiniMax H3) only change steps and CFG
 * and are unaffected.
 */
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
  // And LTX-2's audio stream is guided on its own scale, off one shared unconditional pass
  // (`LTX2Guidance.passes`): audio CFG above 1 consumes the negative prompt even at video CFG 1.
  const audioCfgActive = config.guidance.audioVisible && settings.audioCfgScale !== null && settings.audioCfgScale > 1;
  // An accelerator that declares a guidance triple drives every scale to its identity, which makes
  // the run guidance-free -- the same model the family's distilled *checkpoint* variant is, and that
  // variant declares `usage: 'never'`. Presenting the accelerated model any differently leaves a
  // populated negative prompt on screen that silently stops mattering, and the user never typed the
  // CFG of 1 that disabled it: the toggle did. Families whose accelerator declares no guidance
  // (Wan, MiniMax H3) are unaffected.
  //
  // Not conditioned on the live scale values: with the accelerator on the backend discards them, so
  // there is no state in which raising one makes the negative prompt count again. The scales are
  // not editable in that state either (see `isGuidanceDistilled`), so this cannot strand a value.
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

/**
 * What Expand Prompt starts from for this family: its enhancer and system prompts, with the first frame
 * offered when one is set (only a first frame opens the video it describes).
 */
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
  /** The family's floor for the steps control (validation enforces it too). */
  minSteps: number;
  aspectRatioOptions: readonly VideoAspectRatioId[];
  targetResolutions: readonly VideoTargetResolutionOption[];
  /** Steps for a two-stage preset's refine pass; absent when the base pass's budget serves. */
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
    /** Translation key for the negative prompt's inline help; absent when the field needs none. */
    negativeHelpTextKey?: string;
  };
  /** Ref2VA reference caps; null unless the effective variant has a reference mode. */
  references: { maxVideos: number; maxImages: number; extend?: boolean } | null;
  ui: {
    cfgVisible: boolean;
    cfgLowNoiseVisible: boolean;
    /** LTX-2's per-modality guidance controls; all false on the other families. */
    audioCfgVisible: boolean;
    stgVisible: boolean;
    modalityVisible: boolean;
    fpsVisible: boolean;
    /** False when the schedule is fixed: the steps control shows the count read-only. */
    stepsEditable: boolean;
    /**
     * The extend-mode context control, or null when the mode or family does not have one. `max`
     * moves with the source's own pixels (the join blends at the clip's native resolution), and
     * `newFrames` is what the run actually adds once the crossfade has consumed the context from
     * both halves — the number Frames alone does not reveal.
     */
    extendContext: { value: number; min: number; max: number; step: number; newFrames: number } | null;
    /** The family's distillation fast path, or null when it has none. */
    accelerator: VideoAcceleratorConfig | null;
    /** Steps for the fast path as currently configured, or null when it has none. */
    acceleratorSteps: number | null;
    audioOutput: boolean;
  };
}

/**
 * The range the duration head chooses this run's length from, or null when it does not run.
 *
 * It runs when auto duration is on and the mode leaves the length open, and chooses between the
 * family's shortest clip and the Frames value, which under auto duration is the ceiling the user
 * sized the run for. A ceiling at or under that floor leaves nothing to choose: the head is skipped
 * and the run uses the Frames value as set. The graph and the panel both read this, so the panel's
 * description of the run is the run.
 */
export const getAutoDurationBounds = (
  model: MainModelConfig | undefined,
  settings: VideoSettings
): { maxSeconds: number; minSeconds: number } | null => {
  if (!isAutoDurationActive(settings)) {
    return null;
  }
  const frames = getVideoModelPolicy(model, settings).frames;
  if (frames.kind !== 'grid') {
    return null;
  }
  const timing = getEffectiveVideoTiming(model, settings);

  return ltx2AutoDurationBounds(timing.fps, frames.min, timing.numFrames);
};

export const getVideoModelPolicy = (model: MainModelConfig | undefined, settings: VideoSettings): VideoModelPolicy => {
  const config = getVideoConfig(model);
  // With an accelerator that removes guidance, the backend ignores the scales and the step count
  // outright, so the panel stops offering the controls it would ignore.
  const guidanceDistilled = isGuidanceDistilled(config, settings);
  // Offered only where it means something: LTX-2's extend mode, with a source picked. The ceiling
  // is the source's, so it cannot be computed without one.
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
      // The distilled schedule is a fixed step count the backend clamps to; an editable scrubber
      // here would silently do nothing.
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

/** MiniMax H3 only: the task variant decides which Turbo distillation qualifies. */
export interface FindMiniMaxH3TurboLoraOptions extends FindAcceleratorLoraOptions {
  /** The effective model's variant; ref2va REQUIRES the ref2v-token repack, every other variant excludes it. */
  variant?: string | null;
}

// Delimit both sides with non-alphanumerics: word boundaries miss underscore-separated tokens, while suffix
// matching accepts Slow/Thigh.
const HIGH_NOISE_PATTERN = /(?:^|[^a-z0-9])high(?:[^a-z0-9]|$)/i;
const LOW_NOISE_PATTERN = /(?:^|[^a-z0-9])low(?:[^a-z0-9]|$)/i;

/**
 * Prefer the main's family and assign experts from names. requireFamilyName rejects fallback matches, including
 * models with no known family token.
 */
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

/**
 * Match delimited Turbo names, preferring family names then LightX2V releases with deterministic ties. Exclude
 * Ref2VA-trained releases for FL2VA.
 */
export const findMiniMaxH3TurboLora = (
  models: readonly ModelConfig[],
  { requireFamilyName = false, variant = 'fl2va' }: FindMiniMaxH3TurboLoraOptions = {}
): LoraModelConfig | null => {
  // Family name first (0/2), then the LightX2V generation (0/1): a family-named
  // older repack still beats a LightX2V-named look-alike that omits the family.
  const score = (model: LoraModelConfig): number =>
    (MINIMAX_H3_NAME_PATTERN.test(model.name) ? 0 : 2) + (LIGHTX2V_RELEASE_PATTERN.test(model.name) ? 0 : 1);
  // Within a tier, higher stated schedules rank newer. Ref2VA requires Ref2V-trained releases; other tasks exclude
  // them.
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

/**
 * Return a complete candidate accelerator set or null. Pass the effective model: its H3 task variant determines
 * compatible Turbo releases.
 */
/**
 * The LTX-2.5 step-distillation LoRA in a catalog, or null.
 *
 * Unlike the H3 and Wan families there is one release, and its name states its rank rather than its
 * schedule (`ltx-2.5-22b-distilled-lora-450`), so matching keys on the word that names what it does.
 * `requireFamilyName` exists for the same reason it does elsewhere: on a model switch the panel
 * re-verifies a recorded accelerator, and a bare "distilled" in some other family's LoRA should not
 * satisfy that check.
 */
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

export const findAcceleratorLorasIn = (
  model: MainModelConfig,
  candidates: readonly ModelConfig[],
  options: FindAcceleratorLoraOptions = {}
): LoraModelConfig[] | null => {
  const find = ACCELERATOR_FINDERS[model.base as SupportedVideoBase];

  return find ? find(candidates, model, options) : null;
};

/**
 * How each family locates its own accelerator LoRAs, keyed by base so a new family plugs in without
 * touching the others. Wan's is a *pair* (one per expert); the rest are a single LoRA.
 */
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

/** Resolve steps from an explicit name token, then release overrides, then family defaults. */
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

/** The LoRA models the user currently has switched on in the Concepts list. */
const getEnabledLoraModels = (settings: Pick<VideoSettings, 'loras'>): LoraModelConfig[] =>
  settings.loras.filter((lora) => lora.isEnabled).map((lora) => lora.model);

/** The LoRA models the accelerator toggle recorded, as they stand in the list. */
const getRecordedAcceleratorLoras = (
  settings: Pick<VideoSettings, 'acceleratorEnabled' | 'acceleratorLoraKeys' | 'loras'>
): LoraModelConfig[] => {
  const recorded = new Set(settings.acceleratorLoraKeys);

  return settings.acceleratorEnabled
    ? settings.loras.filter((lora) => recorded.has(lora.model.key)).map((lora) => lora.model)
    : [];
};

/**
 * Accept complete model-family matches or the exact catalog-selected set; the latter permits unusual names
 * installed by the toggle itself.
 */
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

/**
 * Validate only recorded accelerator keys so another enabled candidate cannot displace the user's anchored set
 * through tie-breaking.
 */
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

/**
 * Choose only from the installed catalog; user-list names cannot reliably distinguish distillation from personal
 * style LoRAs.
 */
const findAcceleratorLoraEntries = (model: MainModelConfig, models: readonly ModelConfig[]): GenerateLora[] | null =>
  findAcceleratorLorasIn(model, models)?.map((lora) => ({ isEnabled: true, model: lora, weight: 1 })) ?? null;

export interface AcceleratorToggleResult {
  settings: VideoSettings;
  /** True when enabling was requested but the accelerator LoRA(s) are not installed. */
  missingLoras: boolean;
}

/**
 * Encode acceleration through visible LoRAs and sampling settings so graph compilation needs no hidden toggle
 * behavior.
 */
export const getAcceleratorToggleResult = (
  settings: VideoSettings,
  model: MainModelConfig,
  models: readonly ModelConfig[],
  enabled: boolean
): AcceleratorToggleResult => {
  const config = getVideoConfig(model);
  // Remove exactly the entries a previous toggle added — never a user's own
  // LoRA that happens to share a Lightning/Turbo-style name.
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
        // Only for a family whose accelerator drove these down in the first place. Restoring them
        // unconditionally would reset a user's own guidance tweak on families whose accelerator
        // never touched it.
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
      // A step-distilled model predicts the clean sample directly; leaving the extra guidance
      // scales at their guided defaults samples off the distribution it was fitted to, not merely
      // slower. Families without extra guidance declare none and keep whatever is set.
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

export type AcceleratorLoraChangeOutcome = 'unchanged' | 'switched' | 'disabled';

export interface AcceleratorLoraChangeResult {
  settings: VideoSettings;
  outcome: AcceleratorLoraChangeOutcome;
  /** The LoRA set now driving the fast path — only set when `outcome` is 'switched'. */
  acceleratorLoras: LoraModelConfig[] | null;
}

/**
 * Repair an enabled accelerator from the edited Concepts list: reanchor a complete replacement or disable and
 * restore model defaults. Never enable from a list edit; name heuristics cannot establish that a user LoRA
 * supports distillation.
 */
export const getAcceleratorLoraChangeResult = (
  settings: VideoSettings,
  model: MainModelConfig,
  models: readonly ModelConfig[],
  loras: GenerateLora[]
): AcceleratorLoraChangeResult => {
  const config = getVideoConfig(model);
  const next: VideoSettings = { ...settings, loras };

  // Nothing to repair, and nothing this function is allowed to start.
  if (!settings.acceleratorEnabled) {
    return { acceleratorLoras: null, outcome: 'unchanged', settings: next };
  }

  // The recorded set is still running: leave everything the user tuned alone.
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
      // The same restore the toggle-off path does. Without it, turning the accelerator off by
      // disabling its LoRA leaves the extra scales at the identity values the accelerator wrote --
      // a guided run with three quarters of its guidance silently off, which is the washed-out
      // output this file works to prevent. Only for accelerators that set them in the first place.
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

// A Diffusers Wan main bundles its own VAE and encoder; a GGUF/checkpoint main
// needs them from standalone models or a Diffusers component source.
const getWanComponentSource = (ctx: VideoComponentPolicyContext) =>
  getCompatibleDiffusersComponentSource(ctx.model, ctx.settings.componentSourceModel);

// The UMT5-XXL encoder is shared across Wan families, so any Diffusers source
// supplies it.
const isWanEncoderSatisfied = (ctx: VideoComponentPolicyContext): boolean =>
  ctx.model.format === 'diffusers' || Boolean(getWanComponentSource(ctx));

// The VAE is family-bound (wan_model_loader's source-VAE validation): a source
// only covers it when its TI2V-5B-ness matches the main's.
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

// Low-noise experts must be distinct single-file models of the same variant. Allow unknown variants for backend
// probing.
const isWanLowNoiseExpertCandidate = (candidate: ModelConfig, ctx: VideoComponentPolicyContext): boolean =>
  candidate.type === 'main' &&
  candidate.base === 'wan' &&
  candidate.format !== 'diffusers' &&
  candidate.key !== ctx.model.key &&
  (typeof candidate.variant !== 'string' ||
    typeof ctx.model.variant !== 'string' ||
    candidate.variant === ctx.model.variant);

// TI2V-5B requires 48 VAE channels; A14B requires 16. Allow unspecified channels. Do not await capabilities here:
// early widget sync could otherwise delete and persist a valid stored VAE.
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

  // LTX-2 ships as separate files: the transformer, a component folder (VAEs, vocoder, connectors), and the
  // Gemma-4 encoder that no LTX-2 main carries, so its slot is required rather than an override.
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
        // Optional, and deliberately has no `missingMessage`: without it a run simply uses the
        // frame count the user set, which is what every LTX-2 run did before this model existed.
        modelTypes: ['ltx2_duration_head'],
        required: () => false,
        valueKind: 'component',
      },
    ]);
  }

  // H3 checkpoint mains require a Diffusers component source; components-only sources also require a Qwen3-VL
  // override. Full Diffusers mains bundle both.
  if (model.format === 'diffusers') {
    // Legacy components-only mains remain visible; require the missing encoder while validation requests a
    // single-file transformer main.
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
    ...(model.variant === 'ref2va'
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

// Pruned/full AdaLN shapes differ: require matching checkpoint kinds, leaving unspecified flags to backend
// validation.
const isH3HybridBaseCandidate = (candidate: ModelConfig, ctx: VideoComponentPolicyContext): boolean =>
  candidate.type === 'main' &&
  candidate.base === 'minimax-h3' &&
  candidate.format === 'checkpoint' &&
  candidate.variant === 'fl2va' &&
  candidate.key !== ctx.model.key &&
  (typeof candidate.pruned !== 'boolean' ||
    typeof ctx.model.pruned !== 'boolean' ||
    candidate.pruned === ctx.model.pruned);

/** The H3 Diffusers install a checkpoint main draws its components from, if a valid one is selected. */
const getH3ComponentSource = (ctx: VideoComponentPolicyContext): MainModelConfig | null => {
  const source = ctx.settings.componentSourceModel;

  return source && source.base === 'minimax-h3' && source.format === 'diffusers' ? source : null;
};

// A full Diffusers source carries text-encoder weights; a components-only one
// does not, so the single-file Qwen3-VL override becomes required.
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

/**
 * Expert tags are advisory filename heuristics; explicit wiring remains authoritative and mismatches do not block
 * generation.
 */
export const getWanExpertWiringWarning = (
  model: MainModelConfig | null,
  wanLowNoiseModel: MainModelConfig | null
): WanExpertWiringWarning => {
  // Only single-file A14B mains run the explicit high/low wiring.
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

  // Single expert running the whole schedule: fine for an untagged or
  // high-tagged file, but a low-tagged one is usually the wrong single pick.
  return mainTag === 'low' ? { kind: 'single-low' } : null;
};

/**
 * Prefer full H3 Diffusers installs for checkpoint components; components-only sources also qualify when no full
 * install exists.
 */
const findH3ComponentSource = (models: readonly ModelConfig[]): MainModelConfig | null => {
  const candidates = models.filter(
    (candidate): candidate is ModelConfig & MainModelConfig =>
      candidate.type === 'main' && candidate.base === 'minimax-h3' && candidate.format === 'diffusers'
  );

  return candidates.find((candidate) => !isComponentsOnlyVideoMain(candidate)) ?? candidates[0] ?? null;
};

/** The LTX-2 component folder a generation draws its VAEs, vocoder and connectors from. */
const findLtx2ComponentSource = (models: readonly ModelConfig[]): MainModelConfig | null =>
  models.find(
    (candidate): candidate is ModelConfig & MainModelConfig =>
      candidate.type === 'main' && candidate.base === 'ltx-2' && candidate.format === 'diffusers'
  ) ?? null;

/** The Gemma-4 encoder every LTX-2 generation needs; no main model carries one. */
const findLtx2TextEncoder = (models: readonly ModelConfig[]): ModelConfig | null =>
  models.find((candidate) => candidate.type === 'gemma4_encoder' && candidate.base === 'ltx-2') ?? null;

/**
 * An installed duration head, selected so its Auto duration switch is there to find. Selecting it
 * turns nothing on: the switch itself stays off until the user flips it.
 */
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
    h3HybridBaseModel: modelDefaults.h3HybridBaseModel,
    h3HybridStartBlock: modelDefaults.h3HybridStartBlock,
    h3TextEncoderModel: modelDefaults.h3TextEncoderModel,
    h3TransformerModel: modelDefaults.h3TransformerModel,
    loras: [
      ...settings.loras.filter(
        (lora) => !previousKeys.has(lora.model.key) && !modelDefaults.loras.some((d) => d.model.key === lora.model.key)
      ),
      ...modelDefaults.loras,
    ].map((lora) => (isLoraCompatibleWithModel(lora.model, model) ? lora : { ...lora, isEnabled: false })),
    ltx2TextEncoderModel: modelDefaults.ltx2TextEncoderModel,
    modalityScale: modelDefaults.modalityScale,
    modelKey: model.key,
    numFrames: modelDefaults.numFrames,
    // Reset references with components while preserving valid frame/source conditioning.
    references: modelDefaults.references,
    steps: modelDefaults.steps,
    stgScale: modelDefaults.stgScale,
    targetResolution: modelDefaults.targetResolution,
    vae: modelDefaults.vae,
    wanLowNoiseModel: modelDefaults.wanLowNoiseModel,
    wanT5EncoderModel: modelDefaults.wanT5EncoderModel,
  };
};

/**
 * Setup the user supplied and a switch discards for good: media (with its trims), their own LoRAs, and component
 * choices the previous model would not seed again. Switching back restores none of it, so a UI confirms these.
 */
export type VideoModelSwitchLossKey =
  | 'references'
  | 'initialVideo'
  | 'trimmedInitialVideo'
  | 'conditioningClip'
  | 'firstFrame'
  | 'lastFrame'
  | 'loras'
  | VideoComponentValueKey;

export interface VideoModelSwitchLoss {
  key: VideoModelSwitchLossKey;
  /** How many items go: references or LoRAs; 1 for a single slot. */
  count: number;
}

/**
 * Values the target model constrains or re-defaults. The state does not record whether the user set them, they
 * stay on screen, and each is one number or toggle to re-enter, so they are reported rather than confirmed.
 */
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

/**
 * Use one model-selection transition to reconcile media, sampling constraints, LoRAs, and components, and report
 * what it discards and adjusts. Previewing and applying a switch both call this, so they cannot disagree.
 */
export const getVideoModelSelectionResult = ({
  currentSettings,
  model,
  models,
}: {
  currentSettings: VideoSettings;
  model: MainModelConfig;
  models: readonly ModelConfig[];
}): VideoModelSelectionResult => {
  const config = getVideoConfig(model);
  // Missing modelKey means healing defaults, not user choices; bootstrap selected-model defaults before
  // reconciling seeded media.
  const start = currentSettings.modelKey
    ? currentSettings
    : getVideoSettingsWithModelDefaults(currentSettings, model, models);
  const next: VideoSettings = { ...start, modelKey: model.key };
  const losses: VideoModelSwitchLoss[] = [];
  const adjustments = new Set<VideoModelSwitchAdjustment>();
  const modes = config.modes;
  const previousModel = models.find((entry) => entry.key === currentSettings.modelKey);
  const previousConfig = previousModel && isSupportedVideoModel(previousModel) ? getVideoConfig(previousModel) : null;

  if (next.references.length > 0 && !modes.includes('reference')) {
    // An untouched Initial Video anchor is derived from the initial video and re-derived on the way back; a set
    // trim or conditioning on it is not.
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

  // Snap frame count before deriving reference context so cross-family switches cannot retain an undersized Wan
  // budget.
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
    // The end-frame anchor rides the FLF2V channel whether its partner is a
    // first frame or a source video; alone it needs a dedicated last-frame mode.
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
    // Preserve tuned acceleration when its LoRAs remain compatible. Otherwise reapply the new family's accelerator
    // or restore normal sampling if unavailable.
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

  // The per-modality guidance scales follow the same rule as CFG (Low Noise): a
  // family that does not offer a control carries no value for it, and one that
  // does starts from its own default rather than another family's number.
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
      // Only a value the panel is taking away is a clearing; filling one in for a family that has
      // the control is not. One label for the section rather than one per scale, because they
      // live together under a collapsed "Advanced guidance" heading.
      if (dropped) {
        adjustments.add('advancedGuidance');
      }
    }
  }

  // A family that guides against a list ships it as part of the recipe, exactly like the scales
  // above -- and like them, the fill is from the "this panel carries none" sentinel rather than from
  // any empty value. A variant whose negative prompt is never used hides the field and encodes
  // nothing, so a panel arriving from one holds none to carry, and dev would otherwise guide against
  // an empty string at CFG 3: the one part of the recipe left silently missing. A panel whose
  // previous model cannot be resolved (never seeded, or the model was uninstalled under it) holds
  // none for the same reason, so the automatic re-pick agrees with the manual switch.
  //
  // An empty box on a variant that does show the field is the user's own value and is left alone --
  // that is also what keeps recall's contract that an empty recorded negative prompt does not
  // disturb the panel's. The two cannot be told apart once a panel has passed through a variant that
  // hides the field, so clearing the box, detouring through distilled and coming back restores the
  // list; dev quietly running without it is the worse failure. Filling an empty field in is not a
  // clearing, so it is not reported.
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

  // Steps and CFG have no "carries none" sentinel like the scales above, so a control the previous
  // variant did not offer stands in for one: the user cannot have chosen a number they were never
  // shown, and if it is still that variant's own recommendation it is not one they carried in
  // either. Both conditions are needed. The fixed-schedule variant pins 8 steps at CFG 1, and
  // without this dev arrives holding them -- dev with no guidance at all, which is the recipe that
  // produces washed-out output.
  //
  // Compared against the variant's static recipe, never `getDefaultVideoSettings`: that applies an
  // installed accelerator's numbers, so on a machine with the Lightning LoRAs a Wan panel's
  // "default" is 4/1, and comparing against it would both miss real carry-overs and rewrite a step
  // count the user typed. A control the previous variant *did* show is the user's and is never
  // touched, whatever it holds.
  //
  // Skipped while the accelerator is on: the fast path owns steps and CFG and restores the model's
  // own when it turns off, so its numbers are not a variant's recommendation to compare. An
  // unresolvable previous model leaves both alone -- unlike the blank negative prompt above, where
  // seeding only adds, guessing here would rewrite tuned values on any panel whose model is missing
  // from the catalog. Placed after the fixed-schedule reset so a step count the user did choose is
  // still reported as taken away; adopting a recommendation they never set is not a clearing, and
  // both numbers are on screen.
  //
  // Known gap: a value the user deliberately set to exactly the hiding variant's own default (CFG 1
  // on dev, say) is indistinguishable from one carried in and is replaced. Telling them apart needs
  // a real sentinel on both fields, which is a change to the persisted settings across every family.
  if (!next.acceleratorEnabled && previousConfig) {
    if (!previousConfig.stepsEditable && next.steps === previousConfig.defaults.steps) {
      next.steps = config.defaults.steps;
    }

    if (!previousConfig.cfg.visible && next.cfgScale === previousConfig.defaults.cfgScale) {
      next.cfgScale = config.defaults.cfgScale;
    }
  }

  const compatibleLoras = next.loras.filter((lora) => isLoraCompatibleWithModel(lora.model, model));

  if (compatibleLoras.length !== next.loras.length) {
    // Accelerator LoRAs left with the acceleration reconciliation above; what remains is the user's own.
    losses.push({ count: next.loras.length - compatibleLoras.length, key: 'loras' });
    next.loras = compatibleLoras;
  }

  const policy = getVideoComponentSectionPolicy(model, next);
  const slotsByKey = new Map(policy.slots.map((slotPolicy) => [slotPolicy.key, slotPolicy]));
  // A component the previous model seeds from the catalog is seeded again on the way back, so dropping it loses
  // nothing. Without a resolvable previous model every dropped component counts as the user's choice.
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

  // Auto duration belongs to the head it was turned on with. Once that head is gone, a head filled
  // in below (or picked later) must not bring the switch back on by itself.
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

  // Same for LTX-2, whose encoder slot is required on every model shape.
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

export const getVideoModelAvailabilityReasons = (
  model: MainModelConfig,
  settings: VideoSettings,
  models: readonly ModelConfig[]
): string[] => {
  const reasons: string[] = [];

  if (!hasModelKey(models, model.key, model.type)) {
    reasons.push(`Selected model "${model.name}" is no longer installed.`);
  }

  for (const key of Object.keys(VIDEO_COMPONENT_SETTING_LABELS) as VideoComponentValueKey[]) {
    const value = settings[key];

    if (value && !hasModelKey(models, value.key, value.type)) {
      reasons.push(`${VIDEO_COMPONENT_SETTING_LABELS[key]} "${value.name}" is no longer installed.`);
    }
  }

  for (const lora of settings.loras) {
    if (!hasModelKey(models, lora.model.key, 'lora')) {
      reasons.push(`LoRA "${lora.model.name}" is no longer installed.`);
    }
  }

  return reasons;
};

export const getVideoValidationReasons = (model: MainModelConfig, settings: VideoSettings): string[] => {
  if (!isSupportedVideoModel(model)) {
    return ['Video needs a supported video model before it can be invoked.'];
  }

  // Give repair guidance for stored supported-but-unselectable configurations.
  if (isComponentsOnlyVideoMain(model)) {
    const family = model.base === 'ltx-2' ? 'LTX-2' : 'MiniMax H3';

    return [
      `${model.name} is a components-only install. Select a single-file ${family} transformer as the model; this install then provides its components.`,
    ];
  }
  if (model.base === 'minimax-h3' && model.format === 'diffusers' && model.variant === 'ref2va') {
    return [
      `${model.name} is a Ref2VA folder install, whose transformer weights cannot be folder-loaded. Select a single-file Ref2VA transformer as the model; this install can serve as its Model Components.`,
    ];
  }

  const config = getVideoConfig(model);
  const reasons: string[] = [];
  const mode = resolveVideoMode(settings);
  const referenceOnly = config.modes.length === 1 && config.modes[0] === 'reference';

  if (settings.firstFrameImage && settings.sourceVideo) {
    reasons.push('A first frame and an initial video cannot be combined. Clear one of them.');
  }

  if (settings.conditioningClip && (settings.sourceVideo || settings.references.length > 0)) {
    // Both claim the run's conditioning wholesale, and `resolveVideoMode` would silently drop one
    // rather than run something it cannot express.
    reasons.push('A conditioning clip cannot be combined with an initial video or references. Clear one side.');
  }

  if (
    isConditioningClipExcludingFrames(settings.conditioningClip) &&
    (settings.firstFrameImage || settings.lastFrameImage)
  ) {
    // A clip's picture already fills every frame these would hold. Its soundtrack does not, which is
    // why the frames stay available in that role.
    reasons.push("A conditioning clip's picture cannot be combined with first/last frames. Clear one side.");
  }

  if (settings.references.length > 0 && (settings.firstFrameImage || settings.lastFrameImage)) {
    reasons.push('References cannot be combined with first/last frames. Clear one side.');
  }

  if (settings.references.length > 0 && settings.sourceVideo && !config.references?.extend) {
    reasons.push('References cannot be combined with an initial video on this model. Clear one side.');
  }

  if (!config.modes.includes(mode)) {
    if (referenceOnly && settings.references.length === 0) {
      // Explain the missing reference rather than implying a defective text-to-video model.
      reasons.push('Reference-to-video needs at least one image or video reference.');
    } else {
      reasons.push(`${model.name} does not support ${VIDEO_MODE_DESCRIPTIONS[mode]}.`);
    }
  } else if (mode === 'extend' && settings.lastFrameImage && !config.modes.includes('first-last')) {
    reasons.push(`${model.name} cannot target a destination image while extending a video.`);
  }

  if (mode === 'reference') {
    const caps = config.references;
    const videoCount = settings.references.filter((reference) => reference.kind === 'video').length;
    const imageCount = settings.references.length - videoCount;
    const allAudioOnly =
      settings.references.length > 0 &&
      settings.references.every((reference) => reference.kind === 'video' && reference.conditioning === 'audio');

    if (allAudioOnly) {
      reasons.push(
        'At least one reference must contribute visuals — add an image, or set a video reference to include video.'
      );
    }
    if (caps && videoCount > caps.maxVideos) {
      reasons.push(`At most ${caps.maxVideos} video references are supported.`);
    }
    if (caps && imageCount > caps.maxImages) {
      reasons.push(`At most ${caps.maxImages} image references are supported.`);
    }
    for (const reference of settings.references) {
      if (reference.kind !== 'video') {
        continue;
      }
      if (!Number.isInteger(reference.clip.startFrame) || !Number.isInteger(reference.clip.endFrame)) {
        reasons.push('Reference video trim bounds must be whole frame numbers.');
        break;
      }
      if (
        reference.clip.startFrame < 0 ||
        reference.clip.endFrame > reference.clip.numFrames - 1 ||
        reference.clip.endFrame < reference.clip.startFrame
      ) {
        reasons.push('A reference video trim is outside its clip.');
        break;
      }
    }
  }

  // The effective values rather than the stored ones: a conditioning clip decides both, and the
  // stored numbers it overrides are the user's own -- still theirs once the clip is cleared.
  const timing = getEffectiveVideoTiming(model, settings);

  if (!isValidVideoNumFrames(model, timing.numFrames)) {
    const framesFloor = config.frames.kind === 'grid' ? config.frames.min : (config.frames.choices[0] ?? 0);
    const framesCeiling =
      config.frames.kind === 'grid'
        ? config.frames.max
        : (config.frames.choices[config.frames.choices.length - 1] ?? 0);

    reasons.push(
      timing.numFramesFromClip
        ? `The conditioning clip works out to ${timing.numFrames} frames, outside the ${framesFloor}-${framesCeiling} ${model.name} generates. Use a ${timing.numFrames < framesFloor ? 'longer' : 'shorter'} clip${settings.conditioningClip?.role === 'audio' ? ', or change the frame rate' : ''}.`
        : config.frames.kind === 'grid'
          ? `Frame count must be between ${config.frames.min} and ${config.frames.max} in steps of ${config.frames.step} (${describeVideoFramesGrid(config.frames)}).`
          : `Frame count must be one of the ${model.name} grid values (17·n + 5, ${config.frames.choices[0]}–${config.frames.choices[config.frames.choices.length - 1]}).`
    );
  }

  // fps and steps are integer fields on the backend nodes; a fractional value
  // would fail pydantic coercion at enqueue, so reject it here instead. A rate read off a clip is
  // exempt from the integer rule -- LTX-2's own fps fields are floats, and 29.97 is a real clip.
  if (
    !Number.isFinite(timing.fps) ||
    (!timing.fpsFromClip && !Number.isInteger(timing.fps)) ||
    timing.fps < config.fps.min ||
    timing.fps > config.fps.max
  ) {
    reasons.push(
      timing.fpsFromClip
        ? `The conditioning clip runs at ${timing.fps} fps, outside the ${config.fps.min}-${config.fps.max} fps range ${model.name} supports.`
        : config.fps.editable
          ? `FPS must be a whole number between ${config.fps.min} and ${config.fps.max}.`
          : `${model.name} generates at a fixed ${config.fps.defaultValue} FPS.`
    );
  }

  if (!Number.isInteger(settings.steps) || settings.steps < config.minSteps) {
    reasons.push(`Steps must be a whole number of at least ${config.minSteps}.`);
  }

  // A two-stage preset caps the refine pass at the base pass's budget, and a refine of one step can
  // only re-enter its schedule at the bottom -- which the backend refuses, after the base pass and
  // the upscale have already run. Refused here so the cost is never paid.
  const preset = config.targetResolutions.find((option) => option.id === settings.targetResolution);

  // `ltx2_denoise` refuses a refine pass alongside a held-clean modality: the refine pass re-noises
  // every token, so the conditioning would have to be re-encoded at the refine canvas the way a
  // first frame is. Caught here, before the base pass and the upscale have been paid for.
  if (preset?.stages === 2 && !isTwoStageSupportedForMode(mode)) {
    reasons.push(
      `${VIDEO_MODE_DESCRIPTIONS[mode][0]?.toUpperCase()}${VIDEO_MODE_DESCRIPTIONS[mode].slice(1)} does not run a two-stage target resolution. Pick a single-stage resolution, or clear the conditioning clip.`
    );
  }

  if (preset?.stages === 2 && Number.isInteger(settings.steps) && settings.steps < LTX2_MIN_TWO_STAGE_STEPS) {
    reasons.push(
      `A two-stage target resolution needs at least ${LTX2_MIN_TWO_STAGE_STEPS} steps; the second pass ` +
        `refines what the first produced and cannot run in one.`
    );
  }

  // An LTX-2 continuation replays the front of the source and then joins the two halves by
  // crossfading exactly those frames out of each. Both sides therefore have a floor, and both fail
  // late and confusingly without one: the generated side after the encoders have run, the source
  // side inside the join, after the whole generation.
  if (mode === 'extend' && model.base === 'ltx-2' && settings.sourceVideo) {
    // The user's own value now, not a constant, so every message quotes what they actually set.
    const context = settings.ltx2ExtendContextFrames;

    if (settings.numFrames <= context) {
      reasons.push(
        `A continuation opens with ${context} frames of the source, so anything at or ` +
          `below that would be all replay and no continuation. Raise Frames above ${context}.`
      );
    }

    const affordable = ltx2MaxExtendContextFrames(settings.sourceVideo.width, settings.sourceVideo.height);

    if (affordable === 0) {
      reasons.push(
        `The join blends the held frames of the initial video at its own ` +
          `${settings.sourceVideo.width}x${settings.sourceVideo.height}, which needs more memory than it is ` +
          `allowed even at the smallest context. Use a source at or below about 2560x1440.`
      );
    } else if (context > affordable) {
      // Reachable by loading a recalled or stored setting against a larger source than it was made
      // for; the control itself is bounded, so dragging cannot get here.
      reasons.push(
        `Context Frames is ${context}, but blending that many frames of a ` +
          `${settings.sourceVideo.width}x${settings.sourceVideo.height} source needs more memory than the join ` +
          `is allowed. Lower it to ${affordable} or less.`
      );
    }

    const kept = settings.sourceVideo.endFrame - settings.sourceVideo.startFrame + 1;

    if (kept < context) {
      reasons.push(
        `The join blends ${context} frames out of each half, but the initial video's trim ` +
          `keeps only ${kept}. Keep at least ${context} frames of it, or lower Context Frames.`
      );
    }
  }

  if (config.cfg.visible && (!Number.isFinite(settings.cfgScale) || settings.cfgScale < 1)) {
    reasons.push('CFG must be at least 1.');
  }

  if (config.cfg.lowNoiseVisible && settings.cfgScaleLowNoise !== null && settings.cfgScaleLowNoise < 1) {
    reasons.push('CFG (Low Noise) must be at least 1.');
  }

  // The floors are the denoise node's own `ge`: a value below one fails
  // pydantic validation at enqueue, after the graph has been built.
  const guidanceBounds = [
    { floor: 1, label: 'Audio CFG', value: config.guidance.audioVisible ? settings.audioCfgScale : null },
    { floor: 0, label: 'STG', value: config.guidance.stgVisible ? settings.stgScale : null },
    { floor: 1, label: 'Modality guidance', value: config.guidance.modalityVisible ? settings.modalityScale : null },
  ];

  for (const { floor, label, value } of guidanceBounds) {
    if (value !== null && (!Number.isFinite(value) || value < floor)) {
      reasons.push(`${label} must be at least ${floor}.`);
    }
  }

  if (settings.acceleratorEnabled && !config.accelerator) {
    reasons.push(`${model.name} has no distillation fast path. Turn the accelerator off to generate with it.`);
  }

  // The trim bounds are integer fields on extract_video_range; a fractional
  // persisted value would fail pydantic coercion at enqueue.
  if (
    settings.sourceVideo &&
    (!Number.isInteger(settings.sourceVideo.startFrame) || !Number.isInteger(settings.sourceVideo.endFrame))
  ) {
    reasons.push('The initial video trim bounds must be whole frame numbers.');
  } else if (settings.sourceVideo) {
    const { endFrame, numFrames, startFrame } = settings.sourceVideo;

    // Require valid source bounds and at least two frames for crossfade before encoding.
    if (startFrame < 0 || endFrame > numFrames - 1 || endFrame - startFrame + 1 < MIN_VIDEO_TRIM_FRAMES) {
      reasons.push('The initial video trim must keep at least two frames within the clip.');
    }

    if (numFrames < MIN_VIDEO_TRIM_FRAMES) {
      reasons.push('The initial video is too short to extend.');
    }

    // Wan extension inherits source fps; validate the backend's 1–120 range before expensive denoising.
    if (model.base === 'wan') {
      const inheritedFps = Math.round(settings.sourceVideo.fps);

      if (inheritedFps < 1 || inheritedFps > 120) {
        reasons.push(
          `The initial video's frame rate (${settings.sourceVideo.fps} fps) is outside the 1-120 fps range Wan extension supports.`
        );
      }
    }
  }

  if (!getVideoDimensions(model, settings)) {
    reasons.push(
      model.base === 'minimax-h3'
        ? 'MiniMax H3 supports aspect ratios from 1:4 to 4:1. The conditioning media is outside that range.'
        : 'The conditioning media is too small or degenerate to derive video dimensions from.'
    );
  }

  if (model.base === 'wan') {
    // Report incompatible Wan expert-family LoRAs rather than silently dropping weights that would mismatch tensor
    // shapes.
    for (const lora of settings.loras) {
      if (lora.isEnabled && !isWanLoraTargetingMain(lora.model.variant, model.variant)) {
        reasons.push(`${lora.model.name} targets a different Wan model family than ${model.name}.`);
      }
    }
  }

  const componentPolicy = getVideoComponentSectionPolicy(model, settings);
  reasons.push(...componentPolicy.validate(getVideoComponentPolicyContext(model, settings)));

  return reasons;
};
