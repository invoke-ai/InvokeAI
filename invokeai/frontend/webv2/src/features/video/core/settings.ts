import type { GenerateLora, ImageWithDims } from '@features/generation/contracts';

import {
  DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX,
  DEFAULT_POSITIVE_PROMPT_HEIGHT_PX,
  isLoraModelConfig,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeModelConfig,
  MAX_NEGATIVE_PROMPT_HEIGHT_PX,
  MAX_POSITIVE_PROMPT_HEIGHT_PX,
  MIN_NEGATIVE_PROMPT_HEIGHT_PX,
  MIN_POSITIVE_PROMPT_HEIGHT_PX,
  sanitizeBatchCount,
} from '@features/generation/settings';

import type {
  MiniMaxH3TargetResolution,
  VideoAspectRatioId,
  VideoGenerationMode,
  VideoSettings,
  VideoSourceClip,
  VideoTargetResolution,
  VideoWidgetValues,
  WanTargetResolution,
} from './types';

const isRecord = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object';

const hasFiniteNumber = (record: Record<string, unknown>, key: string): boolean =>
  typeof record[key] === 'number' && Number.isFinite(record[key]);

const getClampedNumber = (record: Record<string, unknown>, key: string, min: number, max: number, fallback: number) =>
  hasFiniteNumber(record, key) ? Math.min(Math.max(record[key] as number, min), max) : fallback;

export const VIDEO_ASPECT_RATIO_IDS: readonly VideoAspectRatioId[] = [
  '21:9',
  '16:9',
  '3:2',
  '4:3',
  '1:1',
  '3:4',
  '2:3',
  '9:16',
  '9:21',
];

export const isVideoAspectRatioId = (value: unknown): value is VideoAspectRatioId =>
  typeof value === 'string' && (VIDEO_ASPECT_RATIO_IDS as readonly string[]).includes(value);

export const WAN_TARGET_RESOLUTIONS: readonly WanTargetResolution[] = ['480p', '720p', '1080p'];
export const MINIMAX_H3_TARGET_RESOLUTIONS: readonly MiniMaxH3TargetResolution[] = ['768 highres', '768 lowres'];

export const isVideoTargetResolution = (value: unknown): value is VideoTargetResolution =>
  typeof value === 'string' &&
  ((WAN_TARGET_RESOLUTIONS as readonly string[]).includes(value) ||
    (MINIMAX_H3_TARGET_RESOLUTIONS as readonly string[]).includes(value));

export const isImageWithDims = (value: unknown): value is ImageWithDims =>
  isRecord(value) &&
  typeof value.image_name === 'string' &&
  hasFiniteNumber(value, 'width') &&
  hasFiniteNumber(value, 'height');

export const isVideoSourceClip = (value: unknown): value is VideoSourceClip =>
  isRecord(value) &&
  typeof value.video_name === 'string' &&
  hasFiniteNumber(value, 'width') &&
  hasFiniteNumber(value, 'height') &&
  hasFiniteNumber(value, 'numFrames') &&
  hasFiniteNumber(value, 'fps') &&
  hasFiniteNumber(value, 'startFrame') &&
  hasFiniteNumber(value, 'endFrame');

const isVideoLora = (value: unknown): value is GenerateLora =>
  isRecord(value) &&
  isLoraModelConfig(value.model) &&
  hasFiniteNumber(value, 'weight') &&
  typeof value.isEnabled === 'boolean';

const getStringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === 'string') : [];

/** Whether every key the accelerator toggle recorded is still present AND enabled in the LoRA list. */
const areAcceleratorLorasPresent = (keys: readonly string[], loras: readonly GenerateLora[]): boolean =>
  keys.length > 0 && keys.every((key) => loras.some((lora) => lora.model.key === key && lora.isEnabled));

/**
 * Which inputs are filled decides the mode; there is no mode selector. A first
 * frame and a source video never coexist (normalization and the setters both
 * enforce it), and a last frame refines whichever mode its partner implies:
 * with a first frame it becomes FLF2V interpolation, with a source video it is
 * the destination the extension should land on.
 */
export const resolveVideoMode = (
  settings: Pick<VideoSettings, 'firstFrameImage' | 'lastFrameImage' | 'sourceVideo'>
): VideoGenerationMode => {
  if (settings.sourceVideo) {
    return 'extend';
  }

  if (settings.firstFrameImage) {
    return settings.lastFrameImage ? 'first-last' : 'first-frame';
  }

  return settings.lastFrameImage ? 'last-frame' : 'txt2vid';
};

/**
 * The fields of `VideoSettings` that describe how the panel is arranged rather
 * than what will be generated. Mirrors `GENERATE_UI_STATE_KEYS`.
 */
export const VIDEO_UI_STATE_KEYS = {
  batchCount: true,
  negativePromptHeightPx: true,
  positivePromptHeightPx: true,
} satisfies Partial<Record<keyof VideoSettings, true>>;

// Model-agnostic fallbacks for healing partial records. They intentionally
// mirror the Wan fallback in the capability matrix, which cannot be imported
// here (videoPolicies imports this module); when a model is known, the
// selection transition re-snaps them to its family anyway.
const SETTINGS_FALLBACKS = {
  aspectRatioId: '16:9',
  cfgScale: 5,
  fps: 16,
  numFrames: 81,
  steps: 40,
  targetResolution: '720p',
} as const;

/**
 * Heals older/partial persisted values without silently clamping invalid user
 * input, upscale-style: any record normalizes field-by-field (so a seeded
 * partial write — e.g. "Send to Video" landing on a never-opened widget —
 * keeps its payload), while range problems stay validation reasons.
 */
export const normalizeVideoSettings = (values: unknown): VideoSettings | null => {
  if (!isRecord(values)) {
    return null;
  }

  const firstFrameImage = isImageWithDims(values.firstFrameImage) ? values.firstFrameImage : null;
  // A first frame and a source video are mutually exclusive; if a stale
  // project somehow holds both, the first frame wins deterministically.
  const sourceVideo = !firstFrameImage && isVideoSourceClip(values.sourceVideo) ? values.sourceVideo : null;
  const loras = Array.isArray(values.loras) ? values.loras.filter(isVideoLora) : [];
  const acceleratorLoraKeys = getStringArray(values.acceleratorLoraKeys);
  // The flag means "the accelerator LoRAs the toggle added are active": if any
  // of them is gone (deleted from Concepts, dropped as incompatible), the flag
  // clears rather than claiming a fast path that has nothing behind it.
  const acceleratorEnabled =
    values.acceleratorEnabled === true && areAcceleratorLorasPresent(acceleratorLoraKeys, loras);

  return {
    aspectRatioId: isVideoAspectRatioId(values.aspectRatioId) ? values.aspectRatioId : SETTINGS_FALLBACKS.aspectRatioId,
    batchCount: sanitizeBatchCount(values.batchCount),
    cfgScale: hasFiniteNumber(values, 'cfgScale') ? (values.cfgScale as number) : SETTINGS_FALLBACKS.cfgScale,
    cfgScaleLowNoise: hasFiniteNumber(values, 'cfgScaleLowNoise') ? (values.cfgScaleLowNoise as number) : null,
    firstFrameImage,
    fps: hasFiniteNumber(values, 'fps') ? (values.fps as number) : SETTINGS_FALLBACKS.fps,
    h3TextEncoderModel: isModelIdentifierConfig(values.h3TextEncoderModel) ? values.h3TextEncoderModel : null,
    h3TransformerModel: isMainModelConfig(values.h3TransformerModel) ? values.h3TransformerModel : null,
    acceleratorEnabled,
    acceleratorLoraKeys: acceleratorEnabled ? acceleratorLoraKeys : [],
    lastFrameImage: isImageWithDims(values.lastFrameImage) ? values.lastFrameImage : null,
    loras,
    modelKey: typeof values.modelKey === 'string' ? values.modelKey : '',
    negativePrompt: typeof values.negativePrompt === 'string' ? values.negativePrompt : '',
    negativePromptEnabled: typeof values.negativePromptEnabled === 'boolean' ? values.negativePromptEnabled : true,
    negativePromptHeightPx: getClampedNumber(
      values,
      'negativePromptHeightPx',
      MIN_NEGATIVE_PROMPT_HEIGHT_PX,
      MAX_NEGATIVE_PROMPT_HEIGHT_PX,
      DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX
    ),
    numFrames: hasFiniteNumber(values, 'numFrames') ? (values.numFrames as number) : SETTINGS_FALLBACKS.numFrames,
    positivePrompt: typeof values.positivePrompt === 'string' ? values.positivePrompt : '',
    positivePromptHeightPx: getClampedNumber(
      values,
      'positivePromptHeightPx',
      MIN_POSITIVE_PROMPT_HEIGHT_PX,
      MAX_POSITIVE_PROMPT_HEIGHT_PX,
      DEFAULT_POSITIVE_PROMPT_HEIGHT_PX
    ),
    seed: hasFiniteNumber(values, 'seed') ? (values.seed as number) : 0,
    shouldRandomizeSeed: typeof values.shouldRandomizeSeed === 'boolean' ? values.shouldRandomizeSeed : true,
    sourceVideo,
    steps: hasFiniteNumber(values, 'steps') ? (values.steps as number) : SETTINGS_FALLBACKS.steps,
    targetResolution: isVideoTargetResolution(values.targetResolution)
      ? values.targetResolution
      : SETTINGS_FALLBACKS.targetResolution,
    vae: isVaeModelConfig(values.vae) ? values.vae : null,
    wanLowNoiseModel: isMainModelConfig(values.wanLowNoiseModel) ? values.wanLowNoiseModel : null,
    wanT5EncoderModel: isModelIdentifierConfig(values.wanT5EncoderModel) ? values.wanT5EncoderModel : null,
    componentSourceModel: isMainModelConfig(values.componentSourceModel) ? values.componentSourceModel : null,
  };
};

export const isVideoSettings = (values: unknown): values is VideoSettings => {
  const normalized = normalizeVideoSettings(values);

  if (!normalized || !isRecord(values)) {
    return false;
  }

  // Strict only over the keys normalize would have to invent.
  return (
    isVideoAspectRatioId(values.aspectRatioId) &&
    isVideoTargetResolution(values.targetResolution) &&
    typeof values.negativePromptEnabled === 'boolean' &&
    typeof values.acceleratorEnabled === 'boolean' &&
    Array.isArray(values.acceleratorLoraKeys) &&
    values.acceleratorLoraKeys.every((key) => typeof key === 'string') &&
    (values.acceleratorEnabled === false
      ? (values.acceleratorLoraKeys as string[]).length === 0
      : Array.isArray(values.loras) &&
        areAcceleratorLorasPresent(values.acceleratorLoraKeys as string[], values.loras.filter(isVideoLora))) &&
    hasFiniteNumber(values, 'negativePromptHeightPx') &&
    hasFiniteNumber(values, 'positivePromptHeightPx') &&
    (values.cfgScaleLowNoise === null || hasFiniteNumber(values, 'cfgScaleLowNoise')) &&
    (values.firstFrameImage === null || isImageWithDims(values.firstFrameImage)) &&
    (values.lastFrameImage === null || isImageWithDims(values.lastFrameImage)) &&
    (values.sourceVideo === null || isVideoSourceClip(values.sourceVideo)) &&
    !(values.firstFrameImage !== null && values.sourceVideo !== null) &&
    Array.isArray(values.loras) &&
    values.loras.every(isVideoLora) &&
    (values.vae === null || isVaeModelConfig(values.vae)) &&
    (values.wanT5EncoderModel === null || isModelIdentifierConfig(values.wanT5EncoderModel)) &&
    (values.wanLowNoiseModel === null || isMainModelConfig(values.wanLowNoiseModel)) &&
    (values.componentSourceModel === null || isMainModelConfig(values.componentSourceModel)) &&
    (values.h3TransformerModel === null || isMainModelConfig(values.h3TransformerModel)) &&
    (values.h3TextEncoderModel === null || isModelIdentifierConfig(values.h3TextEncoderModel))
  );
};

export const normalizeVideoWidgetValues = (values: unknown): VideoWidgetValues | null => {
  const settings = normalizeVideoSettings(values);

  if (!settings || !isRecord(values)) {
    return null;
  }

  return { ...settings, model: isMainModelConfig(values.model) ? values.model : null };
};

export const isVideoWidgetValues = (values: unknown): values is VideoWidgetValues => {
  if (!isVideoSettings(values)) {
    return false;
  }

  const model = (values as unknown as Record<string, unknown>).model;

  return model === null || isMainModelConfig(model);
};

export const cloneVideoWidgetValues = (values: VideoWidgetValues): VideoWidgetValues & Record<string, unknown> => ({
  ...values,
  acceleratorLoraKeys: [...values.acceleratorLoraKeys],
  componentSourceModel: values.componentSourceModel ? { ...values.componentSourceModel } : null,
  firstFrameImage: values.firstFrameImage ? { ...values.firstFrameImage } : null,
  h3TextEncoderModel: values.h3TextEncoderModel ? { ...values.h3TextEncoderModel } : null,
  h3TransformerModel: values.h3TransformerModel ? { ...values.h3TransformerModel } : null,
  lastFrameImage: values.lastFrameImage ? { ...values.lastFrameImage } : null,
  loras: values.loras.map((lora) => ({ ...lora, model: { ...lora.model } })),
  model: values.model ? { ...values.model } : null,
  sourceVideo: values.sourceVideo ? { ...values.sourceVideo } : null,
  vae: values.vae ? { ...values.vae } : null,
  wanLowNoiseModel: values.wanLowNoiseModel ? { ...values.wanLowNoiseModel } : null,
  wanT5EncoderModel: values.wanT5EncoderModel ? { ...values.wanT5EncoderModel } : null,
});

/** Fallback frame rate when a clip's probe did not record one (mirrors extract_video_range). */
export const VIDEO_SOURCE_FALLBACK_FPS = 16;

/**
 * Builds the panel's source-clip record from a gallery video. The frame count
 * is an estimate (duration × fps — the records store no exact count); the
 * backend's extract_video_range resolves authoritative indices at run time.
 * The default trim keeps everything but the final frame (the bundled extend
 * templates' `end_frame: -2`): the extension starts on the trimmed clip's last
 * frame, so keeping the very last one would duplicate it across the seam.
 */
export const createVideoSourceClip = (item: {
  durationSeconds: number;
  fps?: number;
  height: number;
  name: string;
  width: number;
}): VideoSourceClip => {
  const fps = item.fps && Number.isFinite(item.fps) && item.fps > 0 ? item.fps : VIDEO_SOURCE_FALLBACK_FPS;
  const numFrames = Math.max(1, Math.round(item.durationSeconds * fps));

  return {
    // Never below 1: the crossfade join needs at least a two-frame trim.
    endFrame: Math.max(1, numFrames - 2),
    fps,
    height: item.height,
    numFrames,
    startFrame: 0,
    video_name: item.name,
    width: item.width,
  };
};

/** The minimum frames a trim must keep — video_concat's crossfade consumes a 2-frame tail. */
export const MIN_VIDEO_TRIM_FRAMES = 2;

/**
 * Clears conditioning media that no longer exists in the gallery. Returns the
 * input object untouched when nothing changes.
 *
 * Deliberately operates on RAW widget values rather than a normalized
 * snapshot: normalization resolves the first-frame/initial-video exclusion by
 * masking one slot, and a masked reference would silently survive the
 * deletion sweep — a dangling media name waiting to resurface.
 */
export const clearDeletedVideoMedia = <T extends object>(
  values: T,
  removedImageNames: ReadonlySet<string>,
  removedVideoNames: ReadonlySet<string>
): T => {
  const slots = values as { firstFrameImage?: unknown; lastFrameImage?: unknown; sourceVideo?: unknown };
  const clearFirst = isImageWithDims(slots.firstFrameImage) && removedImageNames.has(slots.firstFrameImage.image_name);
  const clearLast = isImageWithDims(slots.lastFrameImage) && removedImageNames.has(slots.lastFrameImage.image_name);
  const clearSource = isVideoSourceClip(slots.sourceVideo) && removedVideoNames.has(slots.sourceVideo.video_name);

  if (!clearFirst && !clearLast && !clearSource) {
    return values;
  }

  // The spread widens the cleared keys to `null`; T itself declares them
  // nullable in every real shape (VideoSettings, raw widget values).
  return {
    ...values,
    ...(clearFirst ? { firstFrameImage: null } : {}),
    ...(clearLast ? { lastFrameImage: null } : {}),
    ...(clearSource ? { sourceVideo: null } : {}),
  } as T;
};
