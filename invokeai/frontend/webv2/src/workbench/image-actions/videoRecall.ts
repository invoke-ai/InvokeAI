import type {
  GenerateLora,
  GenerationModelCatalogItem as ModelConfig,
  MainModelConfig,
  WanLoraTarget,
} from '@features/generation/contracts';
import type {
  VideoAspectRatioId,
  VideoConditioningRole,
  VideoTargetResolution,
  VideoWidgetValues,
} from '@features/video';
import type { TFunction } from 'i18next';

import { isLoraCompatibleWithModel, isLoraModelConfig } from '@features/generation/settings';
import {
  findAcceleratorLorasIn,
  getAcceleratorToggleResult,
  getAcceleratorSteps,
  getEffectiveVideoModel,
  getVideoAspectRatioOptions,
  getVideoDimensions,
  getVideoModelPolicy,
  getVideoModelSelectionResult,
  getVideoTargetResolutionOptions,
  getVideoValidationReasons,
  HOSTED_H3_MODELS,
  isConditioningClipExcludingFrames,
  isSupportedVideoModel,
  isValidVideoNumFrames,
  LTX2_NUM_FRAMES_STEP,
  MINIMAX_H3_FPS,
  MINIMAX_H3_HYBRID_BLOCK_RANGE,
  snapLtx2FramesDown,
  snapVideoNumFrames,
} from '@features/video';
import { SEED_MAX } from '@platform/core/seed';

import {
  mergeOpposingWanLoraTargets,
  readMetadataLoraTarget,
  resolveRecordedModel,
  toRecordedModelRef,
  type RecordedModelPolicy,
  type RecordedModelRef,
} from './recordedModel';

/** Models resolve by installed key, hash, then name/base/type; mode is derived from restored media, never recalled. */

/** Read-only aliases for pre-1.0 record keys (`metadata_version`); never written again. */
const METADATA_KEY_ALIASES: Readonly<Record<string, readonly string[]>> = {
  vae: ['wan_vae_model'],
  wan_guidance_scale_low_noise: ['guidance_scale_low_noise'],
  wan_t5_encoder_model: ['wan_t5_encoder'],
  wan_transformer_low_noise: ['transformer_low_noise'],
};

const VIDEO_GENERATION_MODE_IDS: ReadonlySet<string> = new Set([
  'wan_t2v',
  'wan_i2v',
  'wan_interpolate',
  'wan_extend_video',
  'minimax_h3_t2v',
  'minimax_h3_i2v',
  'minimax_h3_lf2v',
  'minimax_h3_flf2v',
  'minimax_h3_extend_video',
  'minimax_h3_ref2v',
  'ltx2_t2v',
  'ltx2_i2v',
  'ltx2_a2v',
  'ltx2_v2a',
  'ltx2_lf2v',
  'ltx2_flf2v',
  'ltx2_extend_video',
]);

export type VideoRecallKind = 'all' | 'remix' | 'prompts' | 'seed' | 'size';

export interface VideoRecallCapabilities {
  all: boolean;
  remix: boolean;
  prompts: boolean;
  seed: boolean;
  size: boolean;
}

export const EMPTY_VIDEO_RECALL_CAPABILITIES: VideoRecallCapabilities = {
  all: false,
  prompts: false,
  remix: false,
  seed: false,
  size: false,
};

export type VideoRecalledField =
  | 'model'
  | 'prompts'
  | 'seed'
  | 'size'
  | 'frames'
  | 'fps'
  | 'steps'
  | 'cfg'
  | 'loras'
  | 'components'
  | 'extendContext'
  | 'media';

export interface VideoRecallResult {
  fields: VideoRecalledField[];
  values: VideoWidgetValues;
  /** Set when the run's media sized the canvas; `settleMediaSizedRecall` picks the resolution once they hydrate. */
  mediaSizedCanvas?: { height: number; width: number };
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const readKey = (metadata: Record<string, unknown>, key: string): unknown => {
  for (const candidate of [key, ...(METADATA_KEY_ALIASES[key] ?? [])]) {
    if (candidate in metadata) {
      return metadata[candidate];
    }
  }

  return undefined;
};

const getRecord = (metadata: Record<string, unknown>, key: string): Record<string, unknown> | null => {
  const child = readKey(metadata, key);

  return isRecord(child) ? child : null;
};

const getString = (metadata: Record<string, unknown>, key: string): string | null => {
  const value = readKey(metadata, key);

  return typeof value === 'string' ? value : null;
};

const getNullableString = (metadata: Record<string, unknown>, key: string): string | null | undefined => {
  const value = readKey(metadata, key);

  if (value === null) {
    return null;
  }

  return typeof value === 'string' ? value : undefined;
};

const getNumber = (metadata: Record<string, unknown>, key: string): number | null => {
  const value = readKey(metadata, key);

  return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

const getInteger = (metadata: Record<string, unknown>, key: string): number | null => {
  const value = getNumber(metadata, key);

  return value !== null && Number.isInteger(value) ? value : null;
};

const getNumberFrom = (metadata: Record<string, unknown>, key: string, floor: number): number | null => {
  const value = getNumber(metadata, key);

  return value !== null && value >= floor ? value : null;
};

const getIntegerIn = (metadata: Record<string, unknown>, key: string, min: number, max = Infinity): number | null => {
  const value = getInteger(metadata, key);

  return value !== null && value >= min && value <= max ? value : null;
};

/** Video recall applies whatever the record names: a reinstall may change hashes, and the catalog order breaks ties. */
const VIDEO_RECORDED_MODEL_POLICY: RecordedModelPolicy = { ambiguous: 'first', hashMiss: 'matchName' };

const resolveVideoModel = <T extends ModelConfig>(ref: RecordedModelRef | null, models: readonly T[]): T | null => {
  const resolved = ref ? resolveRecordedModel(ref, models, VIDEO_RECORDED_MODEL_POLICY) : null;

  return resolved === 'ambiguous' ? null : resolved;
};

/** Refs resolve only against installed models of the slot's type; untyped refs would name-match anything. */
const VIDEO_COMPONENT_METADATA_KEYS = [
  ['vae', 'vae', 'vae'],
  ['wan_t5_encoder_model', 'wanT5EncoderModel', 'wan_t5_encoder'],
  ['wan_component_source', 'componentSourceModel', 'main'],
  ['wan_transformer_low_noise', 'wanLowNoiseModel', 'main'],
  ['minimax_h3_transformer_model', 'h3TransformerModel', 'main'],
  ['minimax_h3_component_source', 'componentSourceModel', 'main'],
  ['minimax_h3_text_encoder_model', 'h3TextEncoderModel', 'qwen3_vl_encoder'],
  ['minimax_h3_hybrid_base_model', 'h3HybridBaseModel', 'main'],
  ['ltx2_component_source', 'componentSourceModel', 'main'],
  ['ltx2_text_encoder_model', 'ltx2TextEncoderModel', 'gemma4_encoder'],
] as const;

type VideoComponentValuesKey = (typeof VIDEO_COMPONENT_METADATA_KEYS)[number][1];

/** Names only: the executor resolves them against the gallery before they become widget values. */
export interface VideoRecallConditioningClip {
  name: string;
  role: VideoConditioningRole;
}

export interface VideoRecallMediaNames {
  /** LTX-2 whole-modality clip; excludes every other slot except a soundtrack's first and last frames. */
  conditioningClip: VideoRecallConditioningClip | null;
  firstFrameName: string | null;
  lastFrameName: string | null;
  sourceVideoName: string | null;
  sourceVideoTrim: { endFrame: number; startFrame: number } | null;
  /** In conditioning order; the executor drops missing media. */
  references: VideoRecallReferenceName[];
}

export type VideoRecallReferenceName =
  | { kind: 'image'; name: string; detail: string | null }
  | { kind: 'video'; name: string; conditioning: string | null; trim: { endFrame: number; startFrame: number } | null };

/** Recallable values with aliases resolved and malformed values `null`; model refs stay unresolved. */
export interface VideoRecallRecord {
  isVideoGeneration: boolean;
  positivePrompt: string | null;
  /** `null` records the negative prompt switched off; `undefined` records nothing. */
  negativePrompt: string | null | undefined;
  seed: number | null;
  model: RecordedModelRef | null;
  numFrames: number | null;
  steps: number | null;
  cfgScale: number | null;
  /** Absent means the low-noise expert reused the primary CFG. */
  cfgScaleLowNoise: number | null;
  audioCfgScale: number | null;
  stgScale: number | null;
  modalityScale: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  components: readonly { modelType: string; ref: RecordedModelRef; valuesKey: VideoComponentValuesKey }[];
  /** Whether the record carries a `loras` list at all; a partial recall replaces the panel's set only then. */
  hasLoraList: boolean;
  /** The usable entries of `loras`; a missing weight reads as 1. */
  loras: readonly { lowNoiseWeight?: number; model: RecordedModelRef; target?: WanLoraTarget; weight: number }[];
  hybridStartBlock: number | null;
  contextFrames: number | null;
  /** Keyframes and source clip; in extend mode the first frame was extracted from the source, so it is dropped. */
  media: { firstFrameName: string | null; lastFrameName: string | null; sourceVideoName: string | null };
  sourceVideoTrim: { endFrame: number; startFrame: number } | null;
  conditioningClip: VideoRecallConditioningClip | null;
  /** `'empty'` is an explicit request for no references, unlike an absent list or one of malformed entries. */
  referenceList: 'absent' | 'empty' | 'entries';
  references: readonly VideoRecallReferenceName[];
}

const getImageName = (metadata: Record<string, unknown>, key: string): string | null => {
  const field = getRecord(metadata, key);

  return typeof field?.image_name === 'string' ? field.image_name : null;
};

const getVideoName = (metadata: Record<string, unknown>, key: string): string | null => {
  const field = getRecord(metadata, key);

  return typeof field?.video_name === 'string' ? field.video_name : null;
};

const getConditioningClip = (metadata: Record<string, unknown>): VideoRecallConditioningClip | null => {
  const name = getVideoName(metadata, 'ltx2_conditioning_video');
  const role = metadata.ltx2_conditioning_role;

  return name !== null && (role === 'audio' || role === 'video') ? { name, role } : null;
};

const getLoras = (
  metadata: Record<string, unknown>
): { lowNoiseWeight?: number; model: RecordedModelRef; target?: WanLoraTarget; weight: number }[] => {
  if (!Array.isArray(metadata.loras)) {
    return [];
  }

  const loras: { lowNoiseWeight?: number; model: RecordedModelRef; target?: WanLoraTarget; weight: number }[] = [];

  for (const entry of metadata.loras) {
    const model = isRecord(entry) ? toRecordedModelRef(entry.model) : null;

    if (!model) {
      continue;
    }

    const weight = typeof entry.weight === 'number' && Number.isFinite(entry.weight) ? entry.weight : 1;
    const targetFields = isRecord(entry) ? readMetadataLoraTarget(entry, weight) : {};
    const existingIndex = model.key ? loras.findIndex((lora) => lora.model.key === model.key) : -1;

    if (existingIndex !== -1) {
      const merged = mergeOpposingWanLoraTargets(loras[existingIndex]!, { target: targetFields.target, weight });

      if (merged) {
        loras[existingIndex] = merged;
      }
      continue;
    }

    loras.push({ model, weight, ...targetFields });
  }

  return loras;
};

const getReferences = (recorded: unknown[]): VideoRecallReferenceName[] =>
  // Capped at the request maximum (3 videos + 9 images): corrupt metadata must not drive unbounded gallery resolves.
  recorded.slice(0, 12).flatMap((entry): VideoRecallReferenceName[] => {
    if (!isRecord(entry)) {
      return [];
    }
    if (entry.kind === 'image' && typeof entry.image_name === 'string') {
      return [
        { detail: typeof entry.detail === 'string' ? entry.detail : null, kind: 'image', name: entry.image_name },
      ];
    }
    if (entry.kind === 'video' && typeof entry.video_name === 'string') {
      const startFrame =
        typeof entry.start_frame === 'number' && Number.isInteger(entry.start_frame) ? entry.start_frame : null;
      const endFrame =
        typeof entry.end_frame === 'number' && Number.isInteger(entry.end_frame) ? entry.end_frame : null;

      return [
        {
          conditioning: typeof entry.conditioning === 'string' ? entry.conditioning : null,
          kind: 'video',
          name: entry.video_name,
          trim: startFrame !== null && endFrame !== null ? { endFrame, startFrame } : null,
        },
      ];
    }
    return [];
  });

export const parseVideoRecallRecord = (metadata: unknown): VideoRecallRecord | null => {
  if (!isRecord(metadata)) {
    return null;
  }

  const mode = getString(metadata, 'generation_mode');
  const width = getInteger(metadata, 'width');
  const height = getInteger(metadata, 'height');
  const sourceVideoName = getVideoName(metadata, 'source_video');
  const sourceStartFrame = getInteger(metadata, 'source_video_start_frame');
  const sourceEndFrame = getInteger(metadata, 'source_video_end_frame');
  const oldH3Duration =
    mode?.startsWith('minimax_h3_') && VIDEO_GENERATION_MODE_IDS.has(mode)
      ? getIntegerIn(metadata, 'minimax_h3_duration_seconds', 5, 14)
      : null;
  // The legacy 24 fps slider's 14-second stop explicitly selected 345 frames, beyond nearest-grid rounding.
  const legacyH3Frames =
    oldH3Duration === null
      ? null
      : oldH3Duration === 14
        ? 345
        : Math.round((oldH3Duration * MINIMAX_H3_FPS - 5) / 17) * 17 + 5;
  const recordedReferences = Array.isArray(metadata.minimax_h3_references) ? metadata.minimax_h3_references : null;

  return {
    audioCfgScale: getNumberFrom(metadata, 'ltx2_audio_cfg_scale', 1),
    cfgScale: getNumberFrom(metadata, 'cfg_scale', 1),
    cfgScaleLowNoise: getNumberFrom(metadata, 'wan_guidance_scale_low_noise', 1),
    components: VIDEO_COMPONENT_METADATA_KEYS.flatMap(([metadataKey, valuesKey, modelType]) => {
      const ref = toRecordedModelRef(getRecord(metadata, metadataKey));

      return ref ? [{ modelType, ref, valuesKey }] : [];
    }),
    conditioningClip: getConditioningClip(metadata),
    contextFrames: getInteger(metadata, 'ltx2_context_frames'),
    fps: getIntegerIn(metadata, 'fps', 1, 120),
    hasLoraList: Array.isArray(metadata.loras),
    height: height !== null && height > 0 ? height : null,
    hybridStartBlock: getIntegerIn(
      metadata,
      'minimax_h3_hybrid_start_block',
      MINIMAX_H3_HYBRID_BLOCK_RANGE.min,
      MINIMAX_H3_HYBRID_BLOCK_RANGE.max
    ),
    isVideoGeneration: mode !== null && VIDEO_GENERATION_MODE_IDS.has(mode),
    loras: getLoras(metadata),
    media: {
      firstFrameName: sourceVideoName ? null : getImageName(metadata, 'first_frame_image'),
      lastFrameName: getImageName(metadata, 'last_frame_image'),
      sourceVideoName,
    },
    modalityScale: getNumberFrom(metadata, 'ltx2_modality_scale', 1),
    model: toRecordedModelRef(getRecord(metadata, 'model')),
    negativePrompt: getNullableString(metadata, 'negative_prompt'),
    numFrames: getIntegerIn(metadata, 'num_frames', 1) ?? legacyH3Frames,
    positivePrompt: getString(metadata, 'positive_prompt'),
    referenceList: recordedReferences === null ? 'absent' : recordedReferences.length === 0 ? 'empty' : 'entries',
    references: recordedReferences ? getReferences(recordedReferences) : [],
    seed: getIntegerIn(metadata, 'seed', 0, SEED_MAX),
    sourceVideoTrim:
      sourceVideoName && sourceStartFrame !== null && sourceEndFrame !== null
        ? { endFrame: sourceEndFrame, startFrame: sourceStartFrame }
        : null,
    steps: getIntegerIn(metadata, 'steps', 1),
    stgScale: getNumberFrom(metadata, 'ltx2_stg_scale', 0),
    width: width !== null && width > 0 ? width : null,
  };
};

interface VideoSizePreset {
  aspectRatioId: VideoAspectRatioId;
  targetResolution: VideoTargetResolution;
}

/** What a size recall writes: a preset, or only its resolution where conditioning media keep the frame's shape. */
type VideoSizeRecall = VideoSizePreset | Pick<VideoSizePreset, 'targetResolution'>;

/**
 * Presets whose canvas is exactly the recorded size. A media-sized canvas recalls only its target resolution, judged
 * against the media (`settleMediaSizedRecall`). Returns every match: they can differ under held media.
 */
export const getVideoSizeRecalls = (
  model: MainModelConfig,
  width: number | null,
  height: number | null
): VideoSizePreset[] => {
  if (width === null || height === null || width <= 0 || height <= 0) {
    return [];
  }

  const recalls: VideoSizePreset[] = [];

  for (const aspectRatioId of getVideoAspectRatioOptions(model)) {
    for (const option of getVideoTargetResolutionOptions(model)) {
      const derived = getVideoDimensions(model, {
        aspectRatioId,
        conditioningClip: null,
        firstFrameImage: null,
        lastFrameImage: null,
        sourceVideo: null,
        targetResolution: option.id,
      });

      if (derived && derived.width === width && derived.height === height) {
        recalls.push({ aspectRatioId, targetResolution: option.id });
      }
    }
  }

  return recalls;
};

/** Resolve equivalent canvas presets without changing the recorded size; retain the current choice when it fits. */
const getMediaSizedResolution = (
  model: MainModelConfig,
  media: Pick<
    VideoWidgetValues,
    'aspectRatioId' | 'conditioningClip' | 'firstFrameImage' | 'lastFrameImage' | 'sourceVideo' | 'targetResolution'
  >,
  width: number,
  height: number
): Pick<VideoSizePreset, 'targetResolution'> | null => {
  const matches = getVideoTargetResolutionOptions(model).filter((option) => {
    const canvas = getVideoDimensions(model, { ...media, targetResolution: option.id });

    return canvas?.width === width && canvas.height === height;
  });

  const match = matches.find((option) => option.id === media.targetResolution) ?? matches[0];

  return match ? { targetResolution: match.id } : null;
};

const isCanvasFromRecordedMedia = ({ conditioningClip, media }: VideoRecallRecord): boolean =>
  isConditioningClipExcludingFrames(conditioningClip) ||
  media.firstFrameName !== null ||
  media.lastFrameName !== null ||
  media.sourceVideoName !== null;

export type VideoSizeRecallFit =
  /** No model, or neither a preset nor the panel's media at one of its target resolutions reproduces the size. */
  | { outcome: 'unavailable' }
  | { outcome: 'media-unresolvable' }
  | { outcome: 'media-conflict'; effective: { height: number; width: number } }
  /** The size recall would leave the panel unable to run, for this reason it did not already have. */
  | { outcome: 'invalid'; reason: string }
  | { outcome: 'fits'; sizeRecall: VideoSizeRecall };

/** First fitting preset wins, then the single resolution at which the panel's media reach the size. */
export const resolveVideoSizeRecall = (
  values: VideoWidgetValues,
  width: number | null,
  height: number | null
): VideoSizeRecallFit => {
  const { model } = values;

  if (!model) {
    return { outcome: 'unavailable' };
  }

  const existingReasons = new Set(getVideoValidationReasons(model, values));
  const fit = (sizeRecall: VideoSizeRecall): VideoSizeRecallFit => {
    const next = { ...values, ...sizeRecall };
    const effective = getVideoDimensions(model, next);

    if (!effective) {
      return { outcome: 'media-unresolvable' };
    }
    if (effective.width !== width || effective.height !== height) {
      return { effective, outcome: 'media-conflict' };
    }

    const reason = getVideoValidationReasons(model, next).find((candidate) => !existingReasons.has(candidate));

    return reason ? { outcome: 'invalid', reason } : { outcome: 'fits', sizeRecall };
  };
  const presetFits = getVideoSizeRecalls(model, width, height).map(fit);
  const presetFit = presetFits.find((candidate) => candidate.outcome === 'fits');

  if (presetFit) {
    return presetFit;
  }

  const panelCanvas = getVideoDimensions(model, values);
  const mediaSized =
    width !== null && height !== null && panelCanvas && panelCanvas.source !== 'aspect-ratio'
      ? getMediaSizedResolution(model, values, width, height)
      : null;

  return (mediaSized ? fit(mediaSized) : presetFits[0]) ?? { outcome: 'unavailable' };
};

/** Restore acceleration only if recalled LoRAs complete its set at the distilled steps. */
export const deriveAcceleratorRecallState = (
  model: MainModelConfig,
  loras: readonly GenerateLora[],
  steps: number,
  settings: VideoWidgetValues
): Pick<VideoWidgetValues, 'acceleratorEnabled' | 'acceleratorLoraKeys'> => {
  model = getEffectiveVideoModel(model, settings);
  const accelerator = getVideoModelPolicy(model, settings).ui.accelerator;
  const recalled = accelerator
    ? findAcceleratorLorasIn(
        model,
        loras.map((lora) => lora.model),
        { requireFamilyName: true }
      )
    : null;

  return accelerator && recalled && steps === getAcceleratorSteps(accelerator, recalled)
    ? { acceleratorEnabled: true, acceleratorLoraKeys: recalled.map((lora) => lora.key) }
    : { acceleratorEnabled: false, acceleratorLoraKeys: [] };
};

/** Read from the same parsed record the recall applies, so a verb is offered exactly when it writes something. */
const getHostedH3Prompt = (metadata: unknown): string | null => {
  if (
    !isRecord(metadata) ||
    metadata.external_provider !== 'minimax' ||
    !HOSTED_H3_MODELS.some((model) => model === metadata.external_model_id)
  ) {
    return null;
  }
  return getString(metadata, 'positive_prompt');
};

export const getVideoRecallCapabilities = (metadata: unknown): VideoRecallCapabilities => {
  if (getHostedH3Prompt(metadata) !== null) {
    return { ...EMPTY_VIDEO_RECALL_CAPABILITIES, prompts: true };
  }
  const record = parseVideoRecallRecord(metadata);

  if (!record?.isVideoGeneration) {
    return EMPTY_VIDEO_RECALL_CAPABILITIES;
  }

  // A recorded null negative prompt is applied too: it switches the negative prompt off.
  const prompts = record.positivePrompt !== null || record.negativePrompt !== undefined;
  const seed = record.seed !== null;
  const size = record.width !== null && record.height !== null;
  const { media } = record;
  const remix =
    prompts ||
    record.model !== null ||
    record.numFrames !== null ||
    record.steps !== null ||
    record.cfgScale !== null ||
    record.audioCfgScale !== null ||
    record.stgScale !== null ||
    record.modalityScale !== null ||
    record.fps !== null ||
    size ||
    record.components.length > 0 ||
    record.loras.length > 0 ||
    record.hybridStartBlock !== null ||
    record.contextFrames !== null ||
    media.firstFrameName !== null ||
    media.lastFrameName !== null ||
    media.sourceVideoName !== null ||
    record.conditioningClip !== null ||
    record.references.length > 0;

  return { all: remix || seed, prompts, remix, seed, size };
};

const createEmptyMediaNames = (): VideoRecallMediaNames => ({
  conditioningClip: null,
  firstFrameName: null,
  lastFrameName: null,
  references: [],
  sourceVideoName: null,
  sourceVideoTrim: null,
});

/** The media an all/remix recall restores, before the gallery confirms any of it. */
export const getVideoRecallMediaNames = (record: VideoRecallRecord): VideoRecallMediaNames => {
  const { conditioningClip, media, references } = record;
  const mediaNames = createEmptyMediaNames();

  if (conditioningClip) {
    // First and alone: the clip holds a whole modality, except a held soundtrack's first and last frames.
    mediaNames.conditioningClip = conditioningClip;
    if (!isConditioningClipExcludingFrames(conditioningClip)) {
      mediaNames.firstFrameName = media.firstFrameName;
      mediaNames.lastFrameName = media.lastFrameName;
    }
  } else if (references.length > 0) {
    // References replace frame slots, but a recorded source video rides ALONGSIDE: reference-extend appends to it.
    mediaNames.references = [...references];
    if (media.sourceVideoName) {
      mediaNames.sourceVideoName = media.sourceVideoName;
      mediaNames.sourceVideoTrim = record.sourceVideoTrim;
    }
  } else if (media.firstFrameName || media.lastFrameName || media.sourceVideoName) {
    mediaNames.firstFrameName = media.firstFrameName;
    mediaNames.lastFrameName = media.lastFrameName;
    mediaNames.sourceVideoName = media.sourceVideoName;
    mediaNames.sourceVideoTrim = record.sourceVideoTrim;
  }

  return mediaNames;
};

export const buildVideoRecallSettings = ({
  currentValues,
  kind,
  metadata,
  models,
  partial = false,
  requireGenerationMode = true,
}: {
  currentValues: VideoWidgetValues;
  kind: VideoRecallKind;
  metadata: unknown;
  models: readonly ModelConfig[];
  /** Apply only what the record carries (the recall API's default); a full recall clears what it omits. */
  partial?: boolean;
  /** Require video generation metadata; the recall API sends no `generation_mode`. */
  requireGenerationMode?: boolean;
}): (VideoRecallResult & { mediaNames: VideoRecallMediaNames }) | null => {
  const hostedPrompt = getHostedH3Prompt(metadata);
  if (hostedPrompt !== null) {
    return kind === 'prompts'
      ? {
          fields: ['prompts'],
          mediaNames: createEmptyMediaNames(),
          values: { ...currentValues, positivePrompt: hostedPrompt },
        }
      : null;
  }
  const record = parseVideoRecallRecord(metadata);

  if (!record || (requireGenerationMode && !record.isVideoGeneration)) {
    return null;
  }

  const fields: VideoRecalledField[] = [];
  let values: VideoWidgetValues = { ...currentValues };
  // Hold prompts outside values until return so model-default transitions cannot overwrite them.
  let promptPatch: Partial<VideoWidgetValues> | null = null;
  const mediaNames = createEmptyMediaNames();

  if (kind === 'all' || kind === 'prompts' || kind === 'remix') {
    const { negativePrompt, positivePrompt } = record;

    if (positivePrompt !== null || negativePrompt !== undefined) {
      // An empty recorded negative cannot distinguish disabled from empty; preserve the panel's toggle.
      promptPatch = {
        ...(positivePrompt !== null ? { positivePrompt } : {}),
        ...(typeof negativePrompt === 'string' && negativePrompt.length > 0
          ? { negativePrompt, negativePromptEnabled: true }
          : negativePrompt === null
            ? { negativePrompt: '', negativePromptEnabled: false }
            : {}),
      };
      fields.push('prompts');
    }
  }

  if (kind === 'prompts') {
    return fields.length > 0 ? { fields, mediaNames, values: { ...values, ...promptPatch } } : null;
  }

  if (kind === 'all' || kind === 'seed') {
    const { seed } = record;

    if (seed !== null) {
      values = { ...values, seed, seedMode: 'fixed' };
      fields.push('seed');
    }
  }

  if (kind === 'seed') {
    return fields.length > 0 ? { fields, mediaNames, values: { ...values, ...promptPatch } } : null;
  }

  if (kind === 'size') {
    // Fitted to the panel's own model and media: a size recall leaves both alone, so a size they refuse writes nothing.
    const fit = resolveVideoSizeRecall(values, record.width, record.height);

    return fit.outcome === 'fits' ? { fields: ['size'], mediaNames, values: { ...values, ...fit.sizeRecall } } : null;
  }

  const installedModel = resolveVideoModel(record.model, models);
  const recalledModel =
    installedModel && isSupportedVideoModel(installedModel) ? (installedModel as MainModelConfig) : null;
  let model = recalledModel ?? values.model;

  if (recalledModel && recalledModel.key !== values.model?.key) {
    // Family transition first so values snap to the recalled model. Hold its negative prompt back: a seeded default
    // would overrule an empty recording.
    const carriedNegativePrompt = values.negativePrompt;

    values = {
      ...getVideoModelSelectionResult({ currentSettings: values, model: recalledModel, models }).settings,
      model: recalledModel,
      negativePrompt: carriedNegativePrompt,
    };
  }
  if (recalledModel) {
    values.hostedH3 = undefined;
    fields.push('model');
  }

  if (!model) {
    return fields.length > 0 ? { fields, mediaNames, values: { ...values, ...promptPatch } } : null;
  }

  // The existing generation_mode records the effective H3 task, including an override.
  // Restore an override only against the recorded, installed checkpoint.
  const recordedMode = isRecord(metadata) ? getString(metadata, 'generation_mode') : null;
  if (
    model?.base === 'minimax-h3' &&
    model.format === 'checkpoint' &&
    recalledModel &&
    recordedMode?.startsWith('minimax_h3_')
  ) {
    const task = recordedMode === 'minimax_h3_ref2v' ? 'ref2va' : 'fl2va';
    const h3TaskOverride = task !== getEffectiveVideoModel(recalledModel, {}).variant ? task : undefined;
    values = { ...values, model: recalledModel, h3TaskOverride };
  }

  model = getEffectiveVideoModel(model, values);

  const { numFrames } = record;

  if (numFrames !== null) {
    const snapped = isValidVideoNumFrames(model, numFrames) ? numFrames : snapVideoNumFrames(model, numFrames);

    if (snapped !== values.numFrames) {
      values = { ...values, numFrames: snapped };
    }
    // Under auto duration Frames is only a ceiling, so a recalled length would not be the one that runs.
    if (values.autoDuration) {
      values = { ...values, autoDuration: false };
    }
    fields.push('frames');
  }

  // Asked with the accelerator forced OFF: one that removes guidance hides Steps and every scale. A partial record
  // without `loras` keeps the panel's accelerator, so it writes only what that panel shows.
  const lorasRecalled = !partial || record.hasLoraList;
  const policy = getVideoModelPolicy(model, lorasRecalled ? { ...values, acceleratorEnabled: false } : values);
  const { steps } = record;

  // A fixed-schedule checkpoint ignores the step count, so recalling one would show a number the run does not use.
  if (policy.ui.stepsEditable && steps !== null) {
    values = { ...values, steps };
    fields.push('steps');
  }

  const { cfgScale, cfgScaleLowNoise } = record;

  if (policy.ui.cfgVisible && cfgScale !== null) {
    values = {
      ...values,
      cfgScale,
      ...(policy.ui.cfgLowNoiseVisible ? { cfgScaleLowNoise } : {}),
    };
    fields.push('cfg');
  }

  // Per-modality scales ride with CFG; a family without the control must not be handed a number.
  const guidanceRecall = [
    { key: 'audioCfgScale', visible: policy.ui.audioCfgVisible },
    { key: 'stgScale', visible: policy.ui.stgVisible },
    { key: 'modalityScale', visible: policy.ui.modalityVisible },
  ] as const;

  for (const { key, visible } of guidanceRecall) {
    const recalled = record[key];

    if (visible && recalled !== null) {
      values = { ...values, [key]: recalled };

      if (!fields.includes('cfg')) {
        fields.push('cfg');
      }
    }
  }

  const { fps } = record;

  // Wan records delivered fps; extend compilation still inherits clip fps, and H3 fixes it at 24.
  if (policy.fps.editable && fps !== null) {
    if (fps !== values.fps) {
      values = { ...values, fps };
    }
    fields.push('fps');
  }

  const { height, width } = record;
  const [sizeRecall] = getVideoSizeRecalls(model, width, height);
  const mediaSizedCanvas =
    !sizeRecall && width !== null && height !== null && isCanvasFromRecordedMedia(record) ? { height, width } : null;

  if (sizeRecall) {
    values = { ...values, ...sizeRecall };
    fields.push('size');
  }

  // Recall components before deriving acceleration so H3 task detection sees the recorded transformer.
  let componentsRecalled = false;
  let hybridBaseRecalled = false;

  for (const { modelType, ref, valuesKey } of record.components) {
    const installed = resolveVideoModel(
      ref,
      models.filter((candidate) => candidate.type === modelType)
    );

    if (installed) {
      values = { ...values, [valuesKey]: installed };
      componentsRecalled = true;
      hybridBaseRecalled ||= valuesKey === 'h3HybridBaseModel';
    }
  }

  // Restore only an installed recorded hybrid base, else clear it; absent required components keep current picks.
  if (!hybridBaseRecalled && !partial) {
    if (values.h3HybridBaseModel) {
      values = { ...values, h3HybridBaseModel: null };
    }
    componentsRecalled ||= currentValues.h3HybridBaseModel !== null;
  }

  // A full recall restores the start block only with the recorded base; a partial one may set it on the panel's base.
  const { hybridStartBlock } = record;

  if ((hybridBaseRecalled || (partial && values.h3HybridBaseModel !== null)) && hybridStartBlock !== null) {
    values = { ...values, h3HybridStartBlock: hybridStartBlock };
    componentsRecalled = true;
  }

  if (componentsRecalled) {
    fields.push('components');
  }

  // Promote an installed H3 checkpoint from legacy transformer metadata, even if the Diffusers install is gone.
  if (values.h3TransformerModel) {
    const transformer = values.h3TransformerModel;

    if (transformer.type === 'main' && transformer.base === 'minimax-h3' && transformer.format === 'checkpoint') {
      const componentSource =
        model?.base === 'minimax-h3' && model.format === 'diffusers' ? model : values.componentSourceModel;

      model = transformer;
      values = {
        ...values,
        componentSourceModel: componentSource,
        h3TransformerModel: null,
        model,
        modelKey: model.key,
      };
      if (!fields.includes('model')) {
        fields.push('model');
      }
    } else {
      values = { ...values, h3TransformerModel: null };
    }
  }

  const resolvedLoras: GenerateLora[] = [];

  for (const entry of record.loras) {
    const installed = resolveVideoModel(entry.model, models);

    if (!installed || !isLoraModelConfig(installed) || !isLoraCompatibleWithModel(installed, model)) {
      continue;
    }

    const existingIndex = resolvedLoras.findIndex((lora) => lora.model.key === installed.key);

    if (existingIndex !== -1) {
      const merged = mergeOpposingWanLoraTargets(resolvedLoras[existingIndex]!, {
        target: entry.target,
        weight: entry.weight,
      });

      if (merged) {
        resolvedLoras[existingIndex] = merged;
      }
      continue;
    }

    resolvedLoras.push({
      isEnabled: true,
      model: installed,
      weight: entry.weight,
      ...(entry.target !== undefined ? { target: entry.target } : {}),
      ...(entry.lowNoiseWeight !== undefined ? { lowNoiseWeight: entry.lowNoiseWeight } : {}),
    });
  }

  // Reproduce the recorded LoRA set, even empty; a partial record without `loras` leaves the panel's set.
  if (lorasRecalled && (resolvedLoras.length > 0 || values.loras.length > 0)) {
    const accelerator = deriveAcceleratorRecallState(model, resolvedLoras, values.steps, values);

    if (values.acceleratorEnabled && !accelerator.acceleratorEnabled) {
      // Leaving the fast path restores sampling defaults, as the panel toggle does; recorded values stay.
      const defaults = getAcceleratorToggleResult(values, model, models, false).settings;
      values = {
        ...values,
        ...(record.steps !== null ? {} : { steps: defaults.steps }),
        ...(record.cfgScale !== null
          ? {}
          : { cfgScale: defaults.cfgScale, cfgScaleLowNoise: defaults.cfgScaleLowNoise }),
        ...(record.audioCfgScale !== null ? {} : { audioCfgScale: defaults.audioCfgScale }),
        ...(record.stgScale !== null ? {} : { stgScale: defaults.stgScale }),
        ...(record.modalityScale !== null ? {} : { modalityScale: defaults.modalityScale }),
      };
    }
    values = { ...values, loras: resolvedLoras, ...accelerator };
    fields.push('loras');
  }

  // Judged against the original panel: the model transition may already have cleared media.
  const hadMedia = Boolean(
    currentValues.conditioningClip ||
    currentValues.firstFrameImage ||
    currentValues.lastFrameImage ||
    currentValues.sourceVideo ||
    currentValues.references.length > 0
  );

  // A full recall clears media first because media decide the graph family; a partial recall keeps the panel's media.
  if (hadMedia && !partial) {
    values = {
      ...values,
      conditioningClip: null,
      firstFrameImage: null,
      lastFrameImage: null,
      references: [],
      sourceVideo: null,
    };
  }

  const recordedMedia = getVideoRecallMediaNames(record);

  if (
    recordedMedia.conditioningClip ||
    recordedMedia.references.length > 0 ||
    recordedMedia.firstFrameName ||
    recordedMedia.lastFrameName ||
    recordedMedia.sourceVideoName
  ) {
    Object.assign(mediaNames, recordedMedia);
    fields.push('media');
  } else if (hadMedia && !partial) {
    fields.push('media');
  }

  // A partial record's explicitly empty list asks for no references, whatever media it names; an absent one does not.
  if (partial && record.referenceList === 'empty' && values.references.length > 0) {
    values = { ...values, references: [] };
    if (!fields.includes('media')) {
      fields.push('media');
    }
  }

  // Recalled only alongside the source (the output length cannot recover it), snapped like the settings normalizer.
  const { contextFrames } = record;

  if (contextFrames !== null && model?.base === 'ltx-2') {
    values = {
      ...values,
      ltx2ExtendContextFrames: Math.max(1 + LTX2_NUM_FRAMES_STEP, snapLtx2FramesDown(contextFrames)),
    };
    fields.push('extendContext');
  }

  return fields.length > 0
    ? { fields, mediaNames, ...(mediaSizedCanvas ? { mediaSizedCanvas } : {}), values: { ...values, ...promptPatch } }
    : null;
};

/**
 * Settle a media-sized recall against the hydrated media: restore the resolution only when exactly one reproduces the
 * canvas; otherwise leave the panel's resolution and report no size.
 */
export const settleMediaSizedRecall = ({
  fields,
  mediaSizedCanvas,
  values,
}: VideoRecallResult): Pick<VideoRecallResult, 'fields' | 'values'> => {
  const { model } = values;
  const panelCanvas = model ? getVideoDimensions(model, values) : null;
  const sized =
    model && mediaSizedCanvas && panelCanvas && panelCanvas.source !== 'aspect-ratio'
      ? getMediaSizedResolution(model, values, mediaSizedCanvas.width, mediaSizedCanvas.height)
      : null;

  return sized
    ? { fields: fields.includes('size') ? fields : [...fields, 'size'], values: { ...values, ...sized } }
    : { fields, values };
};

export const getVideoRecallTitle = (kind: VideoRecallKind, t: TFunction): string =>
  t(`widgets.gallery.itemActions.recall.videoTitles.${kind}`);

export const getVideoRecallMessage = (fields: readonly VideoRecalledField[], t: TFunction): string =>
  t('widgets.gallery.itemActions.recall.appliedFields', {
    fields: fields.map((field) => t(`widgets.gallery.itemActions.recall.fields.${field}`)),
  });
