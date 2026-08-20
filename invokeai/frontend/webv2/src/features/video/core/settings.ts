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

/**
 * Validate stored widget values and fill in fields that older persisted
 * projects predate. The core fields must be present and well-typed; newer
 * fields fall back to their defaults so old projects keep their prompts and
 * frame counts instead of being reset wholesale.
 */
export const normalizeVideoSettings = (values: unknown): VideoSettings | null => {
  if (!isRecord(values)) {
    return null;
  }

  const isCoreShapeValid =
    typeof values.modelKey === 'string' &&
    typeof values.positivePrompt === 'string' &&
    typeof values.negativePrompt === 'string' &&
    typeof values.shouldRandomizeSeed === 'boolean' &&
    ['numFrames', 'fps', 'steps', 'cfgScale', 'seed'].every((key) => hasFiniteNumber(values, key));

  if (!isCoreShapeValid) {
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
    aspectRatioId: isVideoAspectRatioId(values.aspectRatioId) ? values.aspectRatioId : '16:9',
    batchCount: sanitizeBatchCount(values.batchCount),
    cfgScale: values.cfgScale as number,
    cfgScaleLowNoise: hasFiniteNumber(values, 'cfgScaleLowNoise') ? (values.cfgScaleLowNoise as number) : null,
    firstFrameImage,
    fps: values.fps as number,
    h3TextEncoderModel: isModelIdentifierConfig(values.h3TextEncoderModel) ? values.h3TextEncoderModel : null,
    h3TransformerModel: isMainModelConfig(values.h3TransformerModel) ? values.h3TransformerModel : null,
    acceleratorEnabled,
    acceleratorLoraKeys: acceleratorEnabled ? acceleratorLoraKeys : [],
    lastFrameImage: isImageWithDims(values.lastFrameImage) ? values.lastFrameImage : null,
    loras,
    modelKey: values.modelKey as string,
    negativePrompt: values.negativePrompt as string,
    negativePromptEnabled: typeof values.negativePromptEnabled === 'boolean' ? values.negativePromptEnabled : true,
    negativePromptHeightPx: getClampedNumber(
      values,
      'negativePromptHeightPx',
      MIN_NEGATIVE_PROMPT_HEIGHT_PX,
      MAX_NEGATIVE_PROMPT_HEIGHT_PX,
      DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX
    ),
    numFrames: values.numFrames as number,
    positivePrompt: values.positivePrompt as string,
    positivePromptHeightPx: getClampedNumber(
      values,
      'positivePromptHeightPx',
      MIN_POSITIVE_PROMPT_HEIGHT_PX,
      MAX_POSITIVE_PROMPT_HEIGHT_PX,
      DEFAULT_POSITIVE_PROMPT_HEIGHT_PX
    ),
    seed: values.seed as number,
    shouldRandomizeSeed: values.shouldRandomizeSeed as boolean,
    sourceVideo,
    steps: values.steps as number,
    targetResolution: isVideoTargetResolution(values.targetResolution) ? values.targetResolution : '720p',
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
 */
export const clearDeletedVideoMedia = (
  values: VideoSettings,
  removedImageNames: ReadonlySet<string>,
  removedVideoNames: ReadonlySet<string>
): VideoSettings => {
  const clearFirst = values.firstFrameImage !== null && removedImageNames.has(values.firstFrameImage.image_name);
  const clearLast = values.lastFrameImage !== null && removedImageNames.has(values.lastFrameImage.image_name);
  const clearSource = values.sourceVideo !== null && removedVideoNames.has(values.sourceVideo.video_name);

  if (!clearFirst && !clearLast && !clearSource) {
    return values;
  }

  return {
    ...values,
    ...(clearFirst ? { firstFrameImage: null } : {}),
    ...(clearLast ? { lastFrameImage: null } : {}),
    ...(clearSource ? { sourceVideo: null } : {}),
  };
};
