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
import { isSeedMode } from '@platform/core/seed';

import type {
  Ltx2TargetResolution,
  MiniMaxH3TargetResolution,
  VideoAspectRatioId,
  VideoClipRef,
  VideoConditioningClip,
  VideoConditioningRole,
  VideoGenerationMode,
  VideoReferenceConditioning,
  VideoReferenceImageDetail,
  VideoReferenceItem,
  VideoSettings,
  VideoSourceClip,
  VideoTargetResolution,
  VideoWidgetValues,
  WanTargetResolution,
} from './types';

import { LTX2_EXTEND_CONTEXT_FRAMES, LTX2_NUM_FRAMES_STEP, MINIMAX_H3_FPS, snapLtx2FramesDown } from './dimensions';

const isRecord = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object';

const hasFiniteNumber = (record: Record<string, unknown>, key: string): boolean =>
  typeof record[key] === 'number' && Number.isFinite(record[key]);

const getClampedNumber = (record: Record<string, unknown>, key: string, min: number, max: number, fallback: number) =>
  hasFiniteNumber(record, key) ? Math.min(Math.max(record[key] as number, min), max) : fallback;

/** H3 has blocks 0–49; overlay Ref2VA AdaLN from the chosen start through 49, defaulting to the upper half. */
export const MINIMAX_H3_HYBRID_BLOCK_RANGE = { defaultStart: 25, max: 49, min: 0 } as const;

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
export const LTX2_TARGET_RESOLUTIONS: readonly Ltx2TargetResolution[] = ['512p', '704p', '768p', '1024p', '1536p'];

export const isVideoTargetResolution = (value: unknown): value is VideoTargetResolution =>
  typeof value === 'string' &&
  ((WAN_TARGET_RESOLUTIONS as readonly string[]).includes(value) ||
    (MINIMAX_H3_TARGET_RESOLUTIONS as readonly string[]).includes(value) ||
    (LTX2_TARGET_RESOLUTIONS as readonly string[]).includes(value));

export const isImageWithDims = (value: unknown): value is ImageWithDims =>
  isRecord(value) &&
  typeof value.image_name === 'string' &&
  hasFiniteNumber(value, 'width') &&
  hasFiniteNumber(value, 'height');

export const isVideoClipRef = (value: unknown): value is VideoClipRef =>
  isRecord(value) &&
  typeof value.video_name === 'string' &&
  hasFiniteNumber(value, 'width') &&
  hasFiniteNumber(value, 'height') &&
  hasFiniteNumber(value, 'numFrames') &&
  hasFiniteNumber(value, 'fps');

export const isVideoSourceClip = (value: unknown): value is VideoSourceClip =>
  isRecord(value) &&
  hasFiniteNumber(value, 'startFrame') &&
  hasFiniteNumber(value, 'endFrame') &&
  isVideoClipRef(value);

export const isVideoConditioningRole = (value: unknown): value is VideoConditioningRole =>
  value === 'audio' || value === 'video';

export const isVideoConditioningClip = (value: unknown): value is VideoConditioningClip =>
  isRecord(value) &&
  isVideoClipRef(value.clip) &&
  isVideoConditioningRole(value.role) &&
  typeof value.fpsKnown === 'boolean';

/** Upstream Ref2VA's reference caps (mirrored by the backend's validate_reference_kinds). */
export const VIDEO_REFERENCE_MAX_VIDEOS = 3;
export const VIDEO_REFERENCE_MAX_IMAGES = 9;

/**
 * Default to a short reference sample to limit rows attended at every denoise step; 200 frames is about eight
 * seconds at 24 fps.
 */
export const DEFAULT_REFERENCE_SAMPLE_FRAMES = 200;

/** A sample length clamped to what the clip could ever hold, whatever a control emitted. */
export const clampReferenceSampleFrames = (clip: VideoSourceClip, rawSampleFrames: number): number =>
  Math.min(Math.max(1, Math.round(rawSampleFrames)), Math.max(0, clip.numFrames - 1) + 1);

/**
 * Retain requested sampleFrames separately from clamped bounds so dragging past the end and back restores sample
 * length.
 */
export const referenceSampleFrames = (reference: Extract<VideoReferenceItem, { kind: 'video' }>): number =>
  clampReferenceSampleFrames(
    reference.clip,
    reference.sampleFrames ?? reference.clip.endFrame - reference.clip.startFrame + 1
  );

/**
 * Clamp effective bounds to 0 <= start <= end < numFrames while preserving requested sample length across drags.
 * Anchors use the same movable window policy.
 */
export const slideReferenceSampleWindow = (
  clip: VideoSourceClip,
  rawStart: number,
  requestedSampleFrames?: number
): VideoSourceClip => {
  const maxFrame = Math.max(0, clip.numFrames - 1);
  const sampleFrames = clampReferenceSampleFrames(clip, requestedSampleFrames ?? clip.endFrame - clip.startFrame + 1);
  const startFrame = Math.min(Math.max(0, Math.round(rawStart)), maxFrame);

  return { ...clip, endFrame: Math.min(startFrame + sampleFrames - 1, maxFrame), startFrame };
};

/**
 * Grow from the current start and clamp to the clip; callers persist requested sampleFrames separately from
 * effective bounds.
 */
export const resizeReferenceSampleWindow = (clip: VideoSourceClip, rawSampleFrames: number): VideoSourceClip => {
  const maxFrame = Math.max(0, clip.numFrames - 1);
  const sampleFrames = clampReferenceSampleFrames(clip, rawSampleFrames);
  const startFrame = Math.min(Math.max(0, clip.startFrame), maxFrame);

  return { ...clip, endFrame: Math.min(startFrame + sampleFrames - 1, maxFrame), startFrame };
};

const VIDEO_REFERENCE_CONDITIONINGS = ['video_audio', 'video', 'audio'] as const;

/**
 * Share conditioning recognition with recall so valid recorded choices remain distinct from missing or unusable
 * values.
 */
export const isVideoReferenceConditioning = (value: unknown): value is VideoReferenceConditioning =>
  VIDEO_REFERENCE_CONDITIONINGS.includes(value as (typeof VIDEO_REFERENCE_CONDITIONINGS)[number]);
const VIDEO_REFERENCE_IMAGE_DETAILS = ['max', 'match'] as const;

export const isVideoReferenceItem = (value: unknown): value is VideoReferenceItem => {
  if (!isRecord(value)) {
    return false;
  }
  if (value.kind === 'video') {
    return (
      isVideoSourceClip(value.clip) &&
      VIDEO_REFERENCE_CONDITIONINGS.includes(value.conditioning as (typeof VIDEO_REFERENCE_CONDITIONINGS)[number])
    );
  }
  if (value.kind === 'image') {
    return (
      isImageWithDims(value.image) &&
      VIDEO_REFERENCE_IMAGE_DETAILS.includes(value.detail as (typeof VIDEO_REFERENCE_IMAGE_DETAILS)[number])
    );
  }
  return false;
};

/**
 * Drop invalid/over-cap entries while preserving order. Keep the anchor and oldest video references; discard
 * newest non-anchor overflow. Images retain the front.
 */
const sanitizeVideoReferences = (value: unknown, sourceVideoName?: string): VideoReferenceItem[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  let valid = value.filter((entry) => isVideoReferenceItem(entry));

  // Keep only the last anchor flag to prevent pinning oscillation and multiple cap-exempt entries.
  let flagged = -1;
  for (let index = valid.length - 1; index >= 0; index -= 1) {
    const entry = valid[index]!;

    if (entry.kind === 'video' && entry.fromSourceVideo === true) {
      flagged = index;
      break;
    }
  }
  valid = valid.map((entry, index) =>
    index !== flagged && entry.kind === 'video' && entry.fromSourceVideo === true
      ? // Clear trimOverridden and sample intent with a demoted anchor flag so future adoption can derive a coherent
        // default window.
        { ...entry, fromSourceVideo: false, sampleFrames: undefined, trimOverridden: false }
      : entry
  );

  // Restore omitted recall flags from the last visual same-name reference, preserving earlier user references; an
  // existing flag remains authoritative.
  if (flagged === -1 && sourceVideoName !== undefined) {
    for (let index = valid.length - 1; index >= 0; index -= 1) {
      const entry = valid[index]!;

      if (entry.kind === 'video' && entry.clip.video_name === sourceVideoName && canAnchorReferenceExtend(entry)) {
        valid = valid.map((candidate, candidateIndex) =>
          candidateIndex === index && candidate.kind === 'video' ? { ...candidate, fromSourceVideo: true } : candidate
        );
        break;
      }
    }
  }

  // Pin before trimming overflow so older records with prepended anchors cannot lose their continuity reference.
  valid = pinReferenceExtendAnchor(valid);

  // Keyed by INDEX, not object identity: a hand-edited record can alias the
  // same entry object twice, and an identity Set would drop both copies.
  // With the flag canonicalized to at most one entry above, the flag
  // exemption can never leave the surplus undroppable.
  let videosToDrop = Math.max(0, valid.filter((entry) => entry.kind === 'video').length - VIDEO_REFERENCE_MAX_VIDEOS);
  const dropped = new Set<number>();
  for (let index = valid.length - 1; index >= 0 && videosToDrop > 0; index -= 1) {
    const entry = valid[index]!;

    if (entry.kind === 'video' && entry.fromSourceVideo !== true) {
      dropped.add(index);
      videosToDrop -= 1;
    }
  }

  const result: VideoReferenceItem[] = [];
  let images = 0;
  for (const [index, entry] of valid.entries()) {
    if (entry.kind === 'image') {
      if (images >= VIDEO_REFERENCE_MAX_IMAGES) {
        continue;
      }
      images += 1;
    } else if (dropped.has(index)) {
      continue;
    }
    result.push(entry);
  }
  return result;
};

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
 * Inputs determine mode. First frame/source are exclusive; references win and may coexist with source video for
 * reference extension.
 */
export const resolveVideoMode = (
  settings: Pick<
    VideoSettings,
    'firstFrameImage' | 'lastFrameImage' | 'sourceVideo' | 'references' | 'conditioningClip'
  >
): VideoGenerationMode => {
  if (settings.references.length > 0) {
    return 'reference';
  }

  // Ahead of the frame and clip slots because a conditioning clip excludes them all: it holds one
  // whole modality clean, which is the same mask the other conditioning modes write into.
  if (settings.conditioningClip) {
    return settings.conditioningClip.role === 'audio' ? 'audio-to-video' : 'video-to-audio';
  }

  if (settings.sourceVideo) {
    return 'extend';
  }

  if (settings.firstFrameImage) {
    return settings.lastFrameImage ? 'first-last' : 'first-frame';
  }

  return settings.lastFrameImage ? 'last-frame' : 'txt2vid';
};

/** Panel arrangement fields are UI state rather than generation parameters. */
export const VIDEO_UI_STATE_KEYS = {
  batchCount: true,
  negativePromptHeightPx: true,
  positivePromptHeightPx: true,
} satisfies Partial<Record<keyof VideoSettings, true>>;

// Use model-agnostic Wan fallbacks to avoid a policy import cycle; model selection subsequently snaps to family
// constraints.
const SETTINGS_FALLBACKS = {
  aspectRatioId: '16:9',
  cfgScale: 5,
  fps: 16,
  ltx2ExtendContextFrames: LTX2_EXTEND_CONTEXT_FRAMES,
  numFrames: 81,
  steps: 40,
  targetResolution: '720p',
} as const;

/**
 * Normalize partial records field by field without clamping invalid user values; preserve seeded payloads and
 * report range errors through validation.
 */
export const normalizeVideoSettings = (values: unknown): VideoSettings | null => {
  if (!isRecord(values)) {
    return null;
  }

  // References override stale frame slots; source video may coexist, with model support enforced by validation.
  const references = sanitizeVideoReferences(
    values.references,
    isVideoSourceClip(values.sourceVideo) ? values.sourceVideo.video_name : undefined
  );
  const hasReferences = references.length > 0;
  const firstFrameImage = !hasReferences && isImageWithDims(values.firstFrameImage) ? values.firstFrameImage : null;
  // A first frame and a source video are mutually exclusive; if a stale
  // project somehow holds both, the first frame wins deterministically.
  const sourceVideo = !firstFrameImage && isVideoSourceClip(values.sourceVideo) ? values.sourceVideo : null;
  // One conditioning clip at a time, and never alongside a first frame or an initial video: those
  // condition the same stream this one would hold, and the model samples exactly one modality.
  const conditioningClip =
    !firstFrameImage && !sourceVideo && isVideoConditioningClip(values.conditioningClip)
      ? values.conditioningClip
      : null;
  const loras = Array.isArray(values.loras) ? values.loras.filter(isVideoLora) : [];
  const acceleratorLoraKeys = getStringArray(values.acceleratorLoraKeys);
  // Clear acceleration when any required accelerator LoRA disappears instead of claiming an inactive fast path.
  const acceleratorEnabled =
    values.acceleratorEnabled === true && areAcceleratorLorasPresent(acceleratorLoraKeys, loras);

  return {
    aspectRatioId: isVideoAspectRatioId(values.aspectRatioId) ? values.aspectRatioId : SETTINGS_FALLBACKS.aspectRatioId,
    batchCount: sanitizeBatchCount(values.batchCount),
    cfgScale: hasFiniteNumber(values, 'cfgScale') ? (values.cfgScale as number) : SETTINGS_FALLBACKS.cfgScale,
    // Below 1 the node falls back to the primary CFG, which the widget spells `null`.
    cfgScaleLowNoise:
      hasFiniteNumber(values, 'cfgScaleLowNoise') && (values.cfgScaleLowNoise as number) >= 1
        ? (values.cfgScaleLowNoise as number)
        : null,
    // Null is the healed value for every per-family guidance scale; the model transition fills the family default.
    audioCfgScale: hasFiniteNumber(values, 'audioCfgScale') ? (values.audioCfgScale as number) : null,
    modalityScale: hasFiniteNumber(values, 'modalityScale') ? (values.modalityScale as number) : null,
    stgScale: hasFiniteNumber(values, 'stgScale') ? (values.stgScale as number) : null,
    firstFrameImage,
    fps: hasFiniteNumber(values, 'fps') ? (values.fps as number) : SETTINGS_FALLBACKS.fps,
    h3HybridBaseModel: isMainModelConfig(values.h3HybridBaseModel) ? values.h3HybridBaseModel : null,
    h3HybridStartBlock: Math.round(
      getClampedNumber(
        values,
        'h3HybridStartBlock',
        MINIMAX_H3_HYBRID_BLOCK_RANGE.min,
        MINIMAX_H3_HYBRID_BLOCK_RANGE.max,
        MINIMAX_H3_HYBRID_BLOCK_RANGE.defaultStart
      )
    ),
    h3TextEncoderModel: isModelIdentifierConfig(values.h3TextEncoderModel) ? values.h3TextEncoderModel : null,
    h3TransformerModel: isMainModelConfig(values.h3TransformerModel) ? values.h3TransformerModel : null,
    ltx2TextEncoderModel: isModelIdentifierConfig(values.ltx2TextEncoderModel) ? values.ltx2TextEncoderModel : null,
    acceleratorEnabled,
    acceleratorLoraKeys: acceleratorEnabled ? acceleratorLoraKeys : [],
    lastFrameImage: !hasReferences && isImageWithDims(values.lastFrameImage) ? values.lastFrameImage : null,
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
    // Snapped on the way in: a stored or hand-edited value off the 8k + 1 grid would otherwise
    // reach the node, which snaps it down silently and then reports a different count than the
    // panel shows. Bounds against the source are the validator's job, not this one's -- it has no
    // model or clip to check against.
    ltx2ExtendContextFrames: hasFiniteNumber(values, 'ltx2ExtendContextFrames')
      ? Math.max(1 + LTX2_NUM_FRAMES_STEP, snapLtx2FramesDown(values.ltx2ExtendContextFrames as number))
      : SETTINGS_FALLBACKS.ltx2ExtendContextFrames,
    numFrames: hasFiniteNumber(values, 'numFrames') ? (values.numFrames as number) : SETTINGS_FALLBACKS.numFrames,
    positivePrompt: typeof values.positivePrompt === 'string' ? values.positivePrompt : '',
    positivePromptHeightPx: getClampedNumber(
      values,
      'positivePromptHeightPx',
      MIN_POSITIVE_PROMPT_HEIGHT_PX,
      MAX_POSITIVE_PROMPT_HEIGHT_PX,
      DEFAULT_POSITIVE_PROMPT_HEIGHT_PX
    ),
    references,
    seed: hasFiniteNumber(values, 'seed') ? (values.seed as number) : 0,
    // Values saved before seed modes carry the random toggle instead.
    seedMode: isSeedMode(values.seedMode)
      ? values.seedMode
      : typeof values.shouldRandomizeSeed === 'boolean' && !values.shouldRandomizeSeed
        ? 'fixed'
        : 'random',
    conditioningClip,
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
    isSeedMode(values.seedMode) &&
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
    (values.audioCfgScale === null || hasFiniteNumber(values, 'audioCfgScale')) &&
    (values.stgScale === null || hasFiniteNumber(values, 'stgScale')) &&
    (values.modalityScale === null || hasFiniteNumber(values, 'modalityScale')) &&
    (values.firstFrameImage === null || isImageWithDims(values.firstFrameImage)) &&
    (values.lastFrameImage === null || isImageWithDims(values.lastFrameImage)) &&
    (values.sourceVideo === null || isVideoSourceClip(values.sourceVideo)) &&
    (values.conditioningClip === null || isVideoConditioningClip(values.conditioningClip)) &&
    !(values.firstFrameImage !== null && values.sourceVideo !== null) &&
    Array.isArray(values.references) &&
    values.references.every(isVideoReferenceItem) &&
    !(
      (values.references as unknown[]).length > 0 &&
      (values.firstFrameImage !== null || values.lastFrameImage !== null)
    ) &&
    Array.isArray(values.loras) &&
    values.loras.every(isVideoLora) &&
    (values.vae === null || isVaeModelConfig(values.vae)) &&
    (values.wanT5EncoderModel === null || isModelIdentifierConfig(values.wanT5EncoderModel)) &&
    (values.wanLowNoiseModel === null || isMainModelConfig(values.wanLowNoiseModel)) &&
    (values.componentSourceModel === null || isMainModelConfig(values.componentSourceModel)) &&
    (values.h3TransformerModel === null || isMainModelConfig(values.h3TransformerModel)) &&
    (values.h3TextEncoderModel === null || isModelIdentifierConfig(values.h3TextEncoderModel)) &&
    (values.h3HybridBaseModel === null || isMainModelConfig(values.h3HybridBaseModel)) &&
    (values.ltx2TextEncoderModel === null || isModelIdentifierConfig(values.ltx2TextEncoderModel)) &&
    hasFiniteNumber(values, 'h3HybridStartBlock')
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
  h3HybridBaseModel: values.h3HybridBaseModel ? { ...values.h3HybridBaseModel } : null,
  h3TextEncoderModel: values.h3TextEncoderModel ? { ...values.h3TextEncoderModel } : null,
  h3TransformerModel: values.h3TransformerModel ? { ...values.h3TransformerModel } : null,
  lastFrameImage: values.lastFrameImage ? { ...values.lastFrameImage } : null,
  loras: values.loras.map((lora) => ({ ...lora, model: { ...lora.model } })),
  ltx2TextEncoderModel: values.ltx2TextEncoderModel ? { ...values.ltx2TextEncoderModel } : null,
  model: values.model ? { ...values.model } : null,
  references: values.references.map((reference) =>
    reference.kind === 'video'
      ? { ...reference, clip: { ...reference.clip } }
      : { ...reference, image: { ...reference.image } }
  ),
  conditioningClip: values.conditioningClip
    ? { ...values.conditioningClip, clip: { ...values.conditioningClip.clip } }
    : null,
  sourceVideo: values.sourceVideo ? { ...values.sourceVideo } : null,
  vae: values.vae ? { ...values.vae } : null,
  wanLowNoiseModel: values.wanLowNoiseModel ? { ...values.wanLowNoiseModel } : null,
  wanT5EncoderModel: values.wanT5EncoderModel ? { ...values.wanT5EncoderModel } : null,
});

/** Fallback frame rate when a clip's probe did not record one (mirrors extract_video_range). */
export const VIDEO_SOURCE_FALLBACK_FPS = 16;

/**
 * Estimate frames from duration/fps; backend extraction resolves exact indices. Default trim omits the final frame
 * to avoid duplicating it at the extension seam.
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

/**
 * Audio uploads default to soundtrack-only conditioning. Extension anchors require visual conditioning and use
 * anchorReferenceConditioning instead. Read mediaOrigin from the gallery item; it is not persisted on
 * VideoSourceClip.
 */
export const getDefaultReferenceConditioning = (mediaOrigin: string | null | undefined): VideoReferenceConditioning =>
  mediaOrigin === 'audio_upload' ? 'audio' : 'video_audio';

/**
 * An audio_upload record is a bare soundtrack, so it can only condition the audio side; every other clip
 * defaults to video.
 */
export const getDefaultConditioningRole = (mediaOrigin: string | null | undefined): VideoConditioningRole =>
  mediaOrigin === 'audio_upload' ? 'audio' : 'video';

export const createVideoConditioningClip = (item: {
  durationSeconds: number;
  fps?: number;
  height: number;
  mediaOrigin?: string;
  name: string;
  width: number;
}): VideoConditioningClip => {
  const { endFrame: _endFrame, startFrame: _startFrame, ...clip } = createVideoSourceClip(item);

  return {
    clip,
    fpsKnown: typeof item.fps === 'number' && Number.isFinite(item.fps) && item.fps > 0,
    role: getDefaultConditioningRole(item.mediaOrigin),
  };
};

/**
 * Audio-only references cannot anchor visual continuity. Preserve them as user references and select or append a
 * visual anchor instead.
 */
const canAnchorReferenceExtend = (entry: VideoReferenceItem): boolean =>
  entry.kind === 'video' && entry.conditioning !== 'audio';

/**
 * Promote flagged audio anchors only while deriving their tail window or preserving an explicit override;
 * normalization alone could activate unrelated opening frames.
 */
export const anchorReferenceConditioning = (conditioning: VideoReferenceConditioning): VideoReferenceConditioning =>
  conditioning === 'audio' ? 'video_audio' : conditioning;

/**
 * Mirror build_ref2va_presentation with independent Picture, Video, and Audio counters in attachment order;
 * video_audio advances both relevant counters.
 */
export type VideoReferencePromptLabels = {
  audio: number | null;
  picture: number | null;
  video: number | null;
};

/**
 * Display encoder tokens literally, untranslated, with visual labels first for references carrying both
 * modalities.
 */
export const formatReferencePromptLabels = (labels: VideoReferencePromptLabels): string[] => {
  const formatted: string[] = [];

  if (labels.picture !== null) {
    formatted.push(`<Picture ${labels.picture}>`);
  }
  if (labels.video !== null) {
    formatted.push(`<Video ${labels.video}>`);
  }
  if (labels.audio !== null) {
    formatted.push(`<Audio ${labels.audio}>`);
  }

  return formatted;
};

/** {@link VideoReferencePromptLabels} for every reference, positionally. */
export const referencePromptLabels = (references: readonly VideoReferenceItem[]): VideoReferencePromptLabels[] => {
  const counts = { audio: 0, picture: 0, video: 0 };

  return references.map((reference) => {
    if (reference.kind === 'image') {
      counts.picture += 1;
      return { audio: null, picture: counts.picture, video: null };
    }

    if (reference.conditioning !== 'audio') {
      counts.video += 1;
    }
    if (reference.conditioning !== 'video') {
      counts.audio += 1;
    }

    return {
      audio: reference.conditioning === 'video' ? null : counts.audio,
      picture: null,
      video: reference.conditioning === 'audio' ? null : counts.video,
    };
  });
};

/**
 * Default footage to a bounded head sample. Audio-only references use the whole clip because soundtrack rows are
 * generation-bounded and short windows cut audio early.
 */
export const getDefaultReferenceClip = (
  clip: VideoSourceClip,
  conditioning: VideoReferenceConditioning
): VideoSourceClip => ({
  ...clip,
  endFrame:
    conditioning === 'audio'
      ? Math.max(0, clip.numFrames - 1)
      : Math.max(0, Math.min(DEFAULT_REFERENCE_SAMPLE_FRAMES, clip.numFrames) - 1),
  startFrame: 0,
});

/**
 * A gallery video as a new Ref2VA reference, with the add-path defaults: soundtrack-only conditioning for wrapped
 * audio, and the default sample of the clip. Do not persist mediaOrigin on the clip: project import reuploads under a
 * new name without rederiving it.
 */
export const createVideoReferenceEntry = (item: {
  durationSeconds: number;
  fps?: number;
  height: number;
  mediaOrigin?: string | null;
  name: string;
  width: number;
}): Extract<VideoReferenceItem, { kind: 'video' }> => {
  const conditioning = getDefaultReferenceConditioning(item.mediaOrigin);

  return { clip: getDefaultReferenceClip(createVideoSourceClip(item), conditioning), conditioning, kind: 'video' };
};

/**
 * Default the first image to max detail and later images to generation-matched detail to limit repeatedly attended
 * reference rows; users may override each.
 */
export const getDefaultReferenceImageDetail = (references: VideoReferenceItem[]): VideoReferenceImageDetail =>
  references.some((entry) => entry.kind === 'image') ? 'match' : 'max';

/** The minimum frames a trim must keep — video_concat's crossfade consumes a 2-frame tail. */
export const MIN_VIDEO_TRIM_FRAMES = 2;

/** Inclusive trim ends at (endFrame + 1)/fps to retain its final frame; return null for unusable rates. */
export const videoClipSpanSeconds = (clip: VideoSourceClip): { endSeconds: number; startSeconds: number } | null =>
  Number.isFinite(clip.fps) && clip.fps > 0
    ? { endSeconds: (clip.endFrame + 1) / clip.fps, startSeconds: clip.startFrame / clip.fps }
    : null;

/**
 * Cap tail context at 141 frames of 24-fps material, about 5.9 seconds, further bounded by generated length and
 * converted to source rate.
 */
export const VIDEO_REFERENCE_EXTEND_TAIL_FRAMES = 141;

/**
 * Default to grid-aligned tail context ending at the cutpoint. Account for resampling, generated-length
 * truncation, and 17n+5 snap-down; user trim overrides remain editorial choices.
 */
export const deriveReferenceExtendClip = (sourceVideo: VideoSourceClip, numFrames: number): VideoSourceClip => ({
  ...sourceVideo,
  endFrame: sourceVideo.endFrame,
  // Deliberately unclamped by the Initial Video's START cutpoint: the
  // reference samples the original clip for continuity, independent of which
  // portion the extension keeps.
  startFrame: referenceExtendStartFrame(sourceVideo, numFrames),
});

/**
 * Allow real high-rate containers while bounding fps far below floating-point plateaus that would stop adjustment
 * loops advancing.
 */
const MAX_REFERENCE_SOURCE_FPS = 1e6;

/** `snap_reference_num_frames`: down to the `17n + 5` grid the video VAE encodes whole. */
const snapReferenceFrames = (frames: number): number => Math.max(1, Math.floor((frames - 5) / 17)) * 17 + 5;

/** Source frames -> frames after the backend's 24 fps resample (ffmpeg's fps-filter count). */
const resampledFrameCount = (sourceFrames: number, fps: number): number =>
  Math.floor((sourceFrames * MINIMAX_H3_FPS) / fps + 0.5);

/** The tail window's start index — see `deriveReferenceExtendClip`. */
const referenceExtendStartFrame = (clip: VideoSourceClip, numFrames: number): number => {
  // Use 24 fps for unusable rates; bound extreme rates to prevent nonadvancing floating-point adjustment loops.
  const fps =
    Number.isFinite(clip.fps) && clip.fps > 0 && clip.fps <= MAX_REFERENCE_SOURCE_FPS ? clip.fps : MINIMAX_H3_FPS;
  const requested = Number.isFinite(numFrames)
    ? Math.min(VIDEO_REFERENCE_EXTEND_TAIL_FRAMES, Math.trunc(numFrames))
    : VIDEO_REFERENCE_EXTEND_TAIL_FRAMES;
  // For short clips choose the largest available on-grid budget ending at the cutpoint so snap-down cannot discard
  // seam context.
  const budget = Math.min(
    requested,
    snapReferenceFrames(Math.min(resampledFrameCount(clip.endFrame + 1, fps), requested))
  );

  return Math.max(0, clip.endFrame - (tailSourceFrames(budget, fps) - 1));
};

/**
 * Solve minimum source length from t >= (budget - 0.5)*fps/24, then correct rounding by a bounded few steps. Avoid
 * rate-proportional loops and minimize seam truncation.
 */
const tailSourceFrames = (budget: number, fps: number): number => {
  let tail = Math.max(MIN_VIDEO_TRIM_FRAMES, Math.ceil(((budget - 0.5) * fps) / MINIMAX_H3_FPS));

  while (resampledFrameCount(tail, fps) < budget) {
    tail += 1;
  }
  while (tail > MIN_VIDEO_TRIM_FRAMES && resampledFrameCount(tail - 1, fps) >= budget) {
    tail -= 1;
  }

  return tail;
};

/**
 * Re-derive default windows when generated length changes so temporary input shrinkage cannot ratchet them down.
 * Preserve explicit trim overrides.
 */
export const applyReferenceExtendNumFrames = (
  references: VideoReferenceItem[],
  numFrames: number
): VideoReferenceItem[] => {
  let changed = false;
  const next = references.map((entry) => {
    if (entry.kind !== 'video' || entry.fromSourceVideo !== true || entry.trimOverridden === true) {
      return entry;
    }
    const startFrame = referenceExtendStartFrame(entry.clip, numFrames);

    if (startFrame === entry.clip.startFrame) {
      return entry;
    }
    changed = true;

    // Re-derived, so the user's recorded sample length no longer describes it.
    return { ...entry, clip: { ...entry.clip, startFrame }, sampleFrames: undefined };
  });

  return changed ? next : references;
};

/**
 * Pin the continuity anchor last because generated rotary positions follow the final reference block; preserve
 * identity when already pinned.
 */
export const pinReferenceExtendAnchor = (references: VideoReferenceItem[]): VideoReferenceItem[] => {
  const index = references.findIndex((entry) => entry.kind === 'video' && entry.fromSourceVideo === true);

  if (index < 0 || index === references.length - 1) {
    return references;
  }
  const pinned = references[index]!;

  return [...references.slice(0, index), ...references.slice(index + 1), pinned];
};

/**
 * Mirror setter capacity checks. Existing anchors or adoptable same-clip refs need no new slot; unknown incoming
 * clips conservatively require one.
 */
export const canPlaceReferenceExtendAnchor = (
  references: VideoReferenceItem[],
  videoName: string | undefined,
  maxVideos: number
): boolean => {
  const videos = references.filter(
    (entry): entry is Extract<VideoReferenceItem, { kind: 'video' }> => entry.kind === 'video'
  );

  return (
    videos.some(
      (entry) => entry.fromSourceVideo === true || (videoName !== undefined && entry.clip.video_name === videoName)
    ) || videos.length < maxVideos
  );
};

/**
 * Synchronizes the linked reference with Initial Video. Preserve explicit trims only for the same flagged source;
 * otherwise derive the default. Prefer the flagged anchor, then adopt a matching visual reference, or append if
 * capacity permits. Returning the original list signals capacity refusal. Keep the anchor pinned last.
 */
export const applyReferenceExtendSourceVideo = (
  references: VideoReferenceItem[],
  sourceVideo: VideoSourceClip | null,
  maxVideos: number,
  numFrames: number
): VideoReferenceItem[] => {
  if (!sourceVideo) {
    const kept = references.filter((entry) => !(entry.kind === 'video' && entry.fromSourceVideo === true));

    return kept.length === references.length ? references : kept;
  }

  const linked: VideoReferenceItem = {
    clip: deriveReferenceExtendClip(sourceVideo, numFrames),
    conditioning: 'video_audio',
    fromSourceVideo: true,
    kind: 'video',
  };
  // Prefer the flagged anchor; same-name fallback is only for unflagged recall, preserving the user's other
  // same-clip trims.
  const flaggedIndex = references.findIndex((entry) => entry.kind === 'video' && entry.fromSourceVideo === true);
  const linkedIndex =
    flaggedIndex >= 0
      ? flaggedIndex
      : references.findIndex(
          (entry) =>
            entry.kind === 'video' &&
            entry.clip.video_name === sourceVideo.video_name &&
            // Preserve audio-only same-clip references as soundtracks and append a visual continuity anchor
            // instead.
            canAnchorReferenceExtend(entry)
        );

  if (linkedIndex >= 0) {
    return pinReferenceExtendAnchor(
      references.map((entry, index) => {
        if (index !== linkedIndex || entry.kind !== 'video') {
          return entry;
        }
        const conditioning = anchorReferenceConditioning(entry.conditioning);

        // Only flagged entries can retain anchor overrides; same-name adoption must derive its first anchor
        // window.
        if (
          entry.fromSourceVideo !== true ||
          entry.trimOverridden !== true ||
          entry.clip.video_name !== sourceVideo.video_name
        ) {
          return { ...linked, conditioning };
        }

        return {
          ...linked,
          // Refresh probed geometry/rate while preserving and reclamping the user's trim.
          clip: slideReferenceSampleWindow(
            { ...sourceVideo, endFrame: entry.clip.endFrame, startFrame: entry.clip.startFrame },
            entry.clip.startFrame
          ),
          conditioning,
          sampleFrames: entry.sampleFrames,
          trimOverridden: true,
        };
      })
    );
  }

  const videoCount = references.filter((entry) => entry.kind === 'video').length;

  return videoCount >= maxVideos ? references : [...references, linked];
};

/**
 * The panel patch that makes `sourceVideo` the Initial Video, or clears it. An initial video displaces the first
 * frame and a conditioning clip; on a reference-extend panel it also links its continuity anchor into the
 * references. Returns null when that panel has no reference slot left for the anchor: the clip must not be set
 * without it.
 */
export const getInitialVideoPatch = ({
  maxVideos,
  numFrames,
  referenceExtend,
  references,
  sourceVideo,
}: {
  maxVideos: number;
  numFrames: number;
  referenceExtend: boolean;
  references: VideoReferenceItem[];
  sourceVideo: VideoSourceClip | null;
}): Partial<VideoWidgetValues> | null => {
  const displaced = sourceVideo ? { conditioningClip: null, firstFrameImage: null } : {};

  if (!referenceExtend) {
    return { sourceVideo, ...displaced };
  }

  const linked = applyReferenceExtendSourceVideo(references, sourceVideo, maxVideos, numFrames);

  // Unchanged identity is the capacity refusal; clearing cannot overflow.
  return sourceVideo && linked === references ? null : { references: linked, sourceVideo, ...displaced };
};

/**
 * The panel patch for a new reference list. References displace the frame slots and a conditioning clip, and the
 * initial video too unless the panel extends from it; generation continues from the last reference, so a
 * reference-extend panel keeps its continuity anchor pinned last.
 */
export const getReferencesPatch = ({
  referenceExtend,
  references,
}: {
  referenceExtend: boolean;
  references: VideoReferenceItem[];
}): Partial<VideoWidgetValues> => {
  const next = referenceExtend ? pinReferenceExtendAnchor(references) : references;

  return {
    references: next,
    ...(next.length > 0
      ? {
          conditioningClip: null,
          firstFrameImage: null,
          lastFrameImage: null,
          ...(referenceExtend ? {} : { sourceVideo: null }),
        }
      : {}),
  };
};

/**
 * Clear deleted media from raw values before normalization can mask conflicting slots; preserve identity when
 * unchanged.
 */
export const clearDeletedVideoMedia = <T extends object>(
  values: T,
  removedImageNames: ReadonlySet<string>,
  removedVideoNames: ReadonlySet<string>
): T => {
  const slots = values as {
    conditioningClip?: unknown;
    firstFrameImage?: unknown;
    lastFrameImage?: unknown;
    sourceVideo?: unknown;
    references?: unknown;
  };
  const clearFirst = isImageWithDims(slots.firstFrameImage) && removedImageNames.has(slots.firstFrameImage.image_name);
  const clearLast = isImageWithDims(slots.lastFrameImage) && removedImageNames.has(slots.lastFrameImage.image_name);
  const clearSource = isVideoSourceClip(slots.sourceVideo) && removedVideoNames.has(slots.sourceVideo.video_name);
  const clearConditioning =
    isVideoConditioningClip(slots.conditioningClip) && removedVideoNames.has(slots.conditioningClip.clip.video_name);
  const references = Array.isArray(slots.references) ? slots.references : null;
  const keptReferences = references?.filter(
    (entry) =>
      !isVideoReferenceItem(entry) ||
      (entry.kind === 'video'
        ? !removedVideoNames.has(entry.clip.video_name)
        : !removedImageNames.has(entry.image.image_name))
  );
  const clearReferences = keptReferences !== undefined && keptReferences.length !== references?.length;

  if (!clearFirst && !clearLast && !clearSource && !clearConditioning && !clearReferences) {
    return values;
  }

  // The spread widens the cleared keys to `null`; T itself declares them
  // nullable in every real shape (VideoSettings, raw widget values).
  return {
    ...values,
    ...(clearFirst ? { firstFrameImage: null } : {}),
    ...(clearLast ? { lastFrameImage: null } : {}),
    ...(clearSource ? { sourceVideo: null } : {}),
    ...(clearConditioning ? { conditioningClip: null } : {}),
    ...(clearReferences ? { references: keptReferences } : {}),
  } as T;
};
