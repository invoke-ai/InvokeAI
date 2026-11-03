import type {
  GenerateLora,
  GenerationModelCatalogItem as ModelConfig,
  LoraModelConfig,
  MainModelConfig,
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
  getVideoAspectRatioOptions,
  getVideoDimensions,
  getVideoModelPolicy,
  getVideoModelSelectionResult,
  getVideoTargetResolutionOptions,
  isConditioningClipExcludingFrames,
  isSupportedVideoModel,
  isValidVideoNumFrames,
  LTX2_NUM_FRAMES_STEP,
  MINIMAX_H3_HYBRID_BLOCK_RANGE,
  snapLtx2FramesDown,
  snapVideoNumFrames,
} from '@features/video';
import { SEED_MAX } from '@platform/core/seed';

import {
  resolveRecordedModel,
  toRecordedModelRef,
  type RecordedModelPolicy,
  type RecordedModelRef,
} from './recordedModel';

/**
 * Map video metadata to panel values using installed model key, hash, then name/base/type. Unresolved models
 * retain the current model for validation. Restore media to derive mode rather than recalling mode directly.
 */

/**
 * Canonical record key → the names earlier writers used for the same value.
 * The record format is versioned (`metadata_version`); these are read-only
 * aliases for pre-1.0 records and are never written again.
 */
const METADATA_KEY_ALIASES: Readonly<Record<string, readonly string[]>> = {
  wan_guidance_scale_low_noise: ['guidance_scale_low_noise'],
  wan_t5_encoder_model: ['wan_t5_encoder'],
  wan_transformer_low_noise: ['transformer_low_noise'],
};

/** The generation_mode strings the video graphs stamp; anything else is not video metadata. */
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

export type VideoRecallKind = 'all' | 'remix' | 'prompts' | 'seed';

export interface VideoRecallCapabilities {
  all: boolean;
  remix: boolean;
  prompts: boolean;
  seed: boolean;
}

export const EMPTY_VIDEO_RECALL_CAPABILITIES: VideoRecallCapabilities = {
  all: false,
  prompts: false,
  remix: false,
  seed: false,
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
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

/** The recorded value under `key`, or under the first present alias of it; `undefined` when absent. */
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

/** A recorded number at or above `floor`, or null. */
const getNumberFrom = (metadata: Record<string, unknown>, key: string, floor: number): number | null => {
  const value = getNumber(metadata, key);

  return value !== null && value >= floor ? value : null;
};

/** A recorded integer within `[min, max]`, or null. */
const getIntegerIn = (metadata: Record<string, unknown>, key: string, min: number, max = Infinity): number | null => {
  const value = getInteger(metadata, key);

  return value !== null && value >= min && value <= max ? value : null;
};

/** Video recall applies whatever the record names: a reinstall may change hashes, and the catalog order breaks ties. */
const VIDEO_RECORDED_MODEL_POLICY: RecordedModelPolicy = { ambiguous: 'first', hashMiss: 'matchName' };

/** The installed model a recorded reference names, or null when none does. */
const resolveVideoModel = <T extends ModelConfig>(ref: RecordedModelRef | null, models: readonly T[]): T | null => {
  const resolved = ref ? resolveRecordedModel(ref, models, VIDEO_RECORDED_MODEL_POLICY) : null;

  return resolved === 'ambiguous' ? null : resolved;
};

/**
 * Metadata component slot → widget-values key and the model type the slot holds. The recorded model is resolved
 * against installed models of that type only: a ref without a type would otherwise name-match whatever comes first.
 */
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

/**
 * Media names survive as a separate result field: the executor resolves them
 * against the gallery (existence + dimensions/probe data the metadata does not
 * carry) before they become widget values.
 */
export interface VideoRecallConditioningClip {
  name: string;
  role: VideoConditioningRole;
}

export interface VideoRecallMediaNames {
  /**
   * The LTX-2 whole-modality conditioning clip. It excludes every other slot below, except that a soundtrack
   * recalls alongside the first and last frames that anchored the picture generated for it.
   */
  conditioningClip: VideoRecallConditioningClip | null;
  firstFrameName: string | null;
  lastFrameName: string | null;
  sourceVideoName: string | null;
  /** Restore recorded clip trim so extension starts from the original frame rather than the default. */
  sourceVideoTrim: { endFrame: number; startFrame: number } | null;
  /**
   * The recorded Ref2VA references, in conditioning order. Names and options only — the
   * executor re-resolves each against the gallery and drops missing media, preserving order.
   */
  references: VideoRecallReferenceName[];
}

export type VideoRecallReferenceName =
  | { kind: 'image'; name: string; detail: string | null }
  | { kind: 'video'; name: string; conditioning: string | null; trim: { endFrame: number; startFrame: number } | null };

/**
 * Every value video recall can apply, read once from a record with its key aliases resolved and malformed or
 * out-of-range values dropped (`null`). Model references are as recorded; resolving them against the catalog and the
 * target family's policy is the executor's job, so the same record yields the menu's capabilities and the recall.
 */
export interface VideoRecallRecord {
  /** `generation_mode` is one the video graphs stamp; a video's own record always is. */
  isVideoGeneration: boolean;
  positivePrompt: string | null;
  /** `null` records the negative prompt switched off; `undefined` records nothing. */
  negativePrompt: string | null | undefined;
  seed: number | null;
  model: RecordedModelRef | null;
  /** A positive count; the target model may still snap it. */
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
  /** The recorded component slots with a usable reference, in `VIDEO_COMPONENT_METADATA_KEYS` order. */
  components: readonly { modelType: string; ref: RecordedModelRef; valuesKey: VideoComponentValuesKey }[];
  /** Whether the record carries a `loras` list at all; a partial recall replaces the panel's set only then. */
  hasLoraList: boolean;
  /** The usable entries of `loras`; a missing weight reads as 1. */
  loras: readonly { model: RecordedModelRef; weight: number }[];
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

/** The whole-modality conditioning clip, from the LTX-2 extras the graph records. */
const getConditioningClip = (metadata: Record<string, unknown>): VideoRecallConditioningClip | null => {
  const name = getVideoName(metadata, 'ltx2_conditioning_video');
  const role = metadata.ltx2_conditioning_role;

  return name !== null && (role === 'audio' || role === 'video') ? { name, role } : null;
};

const getLoras = (metadata: Record<string, unknown>): { model: RecordedModelRef; weight: number }[] =>
  Array.isArray(metadata.loras)
    ? metadata.loras.flatMap((entry) => {
        const model = isRecord(entry) ? toRecordedModelRef(entry.model) : null;

        if (!model) {
          return [];
        }

        const weight = typeof entry.weight === 'number' && Number.isFinite(entry.weight) ? entry.weight : 1;

        return [{ model, weight }];
      })
    : [];

/** Tolerant parser for the `minimax_h3_references` metadata extra: skips malformed entries, keeps order. */
const getReferences = (recorded: unknown[]): VideoRecallReferenceName[] =>
  // Cap at the request maximum (3 videos + 9 images): corrupt metadata must not drive an
  // unbounded chain of gallery resolves in the executor.
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

/** The recall-relevant values of a metadata record, or null when it is not a record. */
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
    numFrames: getIntegerIn(metadata, 'num_frames', 1),
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

/**
 * Recall aspect/resolution presets only on exact dimension matches; other dimensions came from conditioning media
 * and remain media-controlled.
 */
export const getVideoSizeRecall = (
  model: MainModelConfig,
  width: number | null,
  height: number | null
): { aspectRatioId: VideoAspectRatioId; targetResolution: VideoTargetResolution } | null => {
  if (width === null || height === null || width <= 0 || height <= 0) {
    return null;
  }

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
        return { aspectRatioId, targetResolution: option.id };
      }
    }
  }

  return null;
};

/**
 * Restore acceleration only when recalled LoRAs form a complete set at its distilled step count; derive from
 * recorded keys, not today's catalog preference.
 */
export const deriveAcceleratorRecallState = (
  model: MainModelConfig,
  loras: readonly GenerateLora[],
  steps: number,
  settings: VideoWidgetValues
): Pick<VideoWidgetValues, 'acceleratorEnabled' | 'acceleratorLoraKeys'> => {
  // Promote legacy H3 transformer overrides before deriving task-specific accelerators: fl2va and ref2v use
  // different sets and steps.
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

/**
 * What the record carries for each verb, independent of the catalog: execution still validates against the panel's
 * model. Read from the same parsed record the recall applies, so a verb is offered exactly when the record names
 * something it would write.
 */
export const getVideoRecallCapabilities = (metadata: unknown): VideoRecallCapabilities => {
  const record = parseVideoRecallRecord(metadata);

  if (!record?.isVideoGeneration) {
    return EMPTY_VIDEO_RECALL_CAPABILITIES;
  }

  // A recorded null negative prompt is applied too: it switches the negative prompt off.
  const prompts = record.positivePrompt !== null || record.negativePrompt !== undefined;
  const seed = record.seed !== null;
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
    (record.width !== null && record.height !== null) ||
    record.components.length > 0 ||
    record.loras.length > 0 ||
    record.hybridStartBlock !== null ||
    record.contextFrames !== null ||
    media.firstFrameName !== null ||
    media.lastFrameName !== null ||
    media.sourceVideoName !== null ||
    record.conditioningClip !== null ||
    record.references.length > 0;

  return { all: remix || seed, prompts, remix, seed };
};

const createEmptyMediaNames = (): VideoRecallMediaNames => ({
  conditioningClip: null,
  firstFrameName: null,
  lastFrameName: null,
  references: [],
  sourceVideoName: null,
  sourceVideoTrim: null,
});

/**
 * The media an all/remix recall of the record restores, before the gallery confirms any of it. The executor reads
 * them ahead of the panel snapshot; `buildVideoRecallSettings` carries the same selection in its result.
 */
export const getVideoRecallMediaNames = (record: VideoRecallRecord): VideoRecallMediaNames => {
  const { conditioningClip, media, references } = record;
  const mediaNames = createEmptyMediaNames();

  if (conditioningClip) {
    // First, and alone but for frames: the clip holds a whole modality clean, so no other slot could
    // have been filled on the run being recalled -- except a held soundtrack's first and last frames.
    mediaNames.conditioningClip = conditioningClip;
    if (!isConditioningClipExcludingFrames(conditioningClip)) {
      mediaNames.firstFrameName = media.firstFrameName;
      mediaNames.lastFrameName = media.lastFrameName;
    }
  } else if (references.length > 0) {
    // References replace the frame slots, but a recorded source video rides
    // ALONGSIDE them: Ref2VA reference-extend appends the new clip to it.
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
  /**
   * Apply only what the record carries, leaving everything it omits as the panel has it — the external recall
   * API's default. A full recall instead reproduces the run: it clears LoRAs, media and the hybrid base the record
   * does not name.
   */
  partial?: boolean;
  /**
   * Whether the record must be video generation metadata. A video's own record always is; the external recall API
   * sends the same keys without a `generation_mode`.
   */
  requireGenerationMode?: boolean;
}): (VideoRecallResult & { mediaNames: VideoRecallMediaNames }) | null => {
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

  // all / remix from here on.
  const installedModel = resolveVideoModel(record.model, models);
  const recalledModel =
    installedModel && isSupportedVideoModel(installedModel) ? (installedModel as MainModelConfig) : null;
  let model = recalledModel ?? values.model;

  if (recalledModel && recalledModel.key !== values.model?.key) {
    // The canonical family transition first, so frames/fps/resolution snap to
    // the recalled model before its recorded values land on top. Its negative
    // prompt is held back: what the clip recorded is authoritative, and an
    // empty recording deliberately leaves the panel's own alone (below), which
    // a family default seeded on the way in would silently overrule -- the
    // recalled clip would then be re-run against a list it never used.
    const carriedNegativePrompt = values.negativePrompt;

    values = {
      ...getVideoModelSelectionResult({ currentSettings: values, model: recalledModel, models }).settings,
      model: recalledModel,
      negativePrompt: carriedNegativePrompt,
    };
    fields.push('model');
  } else if (recalledModel) {
    fields.push('model');
  }

  if (!model) {
    // Without any model, nothing below can validate; prompts/seed may still
    // have been recalled.
    return fields.length > 0 ? { fields, mediaNames, values: { ...values, ...promptPatch } } : null;
  }

  const { numFrames } = record;

  if (numFrames !== null) {
    const snapped = isValidVideoNumFrames(model, numFrames) ? numFrames : snapVideoNumFrames(model, numFrames);

    if (snapped !== values.numFrames) {
      values = { ...values, numFrames: snapped };
    }
    // Under auto duration the Frames value is only a ceiling the duration head chooses beneath, so a
    // recalled length would not be the length that runs. Recall reproduces the clip it came from.
    if (values.autoDuration) {
      values = { ...values, autoDuration: false };
    }
    fields.push('frames');
  }

  // Asked with the accelerator forced OFF, not as the panel currently stands. An accelerator that
  // removes guidance hides Steps and every guidance scale, and this policy decides which of them
  // recall is allowed to write -- so recalling an ordinary clip into a panel that happens to have
  // the accelerator on would drop them all and silently leave the accelerator's values in place,
  // showing numbers the recalled clip never used. The accelerator's own state is derived further
  // down from the recalled LoRA set, which overwrites this anyway.
  // A partial record without `loras` leaves the accelerator as the panel has it, so it may write only the controls
  // that panel shows; everything else re-derives the accelerator from the recalled LoRAs further down.
  const lorasRecalled = !partial || record.hasLoraList;
  const policy = getVideoModelPolicy(model, lorasRecalled ? { ...values, acceleratorEnabled: false } : values);
  const { steps } = record;

  // A fixed-schedule checkpoint ignores whatever step count reaches it, so recalling one would
  // leave a disabled control showing a number the run will not use — and re-record it next time.
  if (policy.ui.stepsEditable && steps !== null) {
    values = { ...values, steps };
    fields.push('steps');
  }

  const { cfgScale, cfgScaleLowNoise } = record;

  if (policy.ui.cfgVisible && cfgScale !== null) {
    values = {
      ...values,
      cfgScale,
      // The graph records the key only when the setting was non-null; absence
      // means "reuse the primary CFG", so restore that exact semantics.
      ...(policy.ui.cfgLowNoiseVisible ? { cfgScaleLowNoise } : {}),
    };
    fields.push('cfg');
  }

  // The per-modality scales ride with CFG: they are the same run's guidance,
  // and a family that does not offer a control must not be handed a number.
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

  const sizeRecall = getVideoSizeRecall(model, record.width, record.height);

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

  // Restore only an installed recorded hybrid base; otherwise clear it and report against the original panel.
  // Required source/text components retain current picks when absent.
  if (!hybridBaseRecalled && !partial) {
    if (values.h3HybridBaseModel) {
      values = { ...values, h3HybridBaseModel: null };
    }
    componentsRecalled ||= currentValues.h3HybridBaseModel !== null;
  }

  // The hybrid's start block belongs to the recorded base: a full recall restores it only with that base, never
  // onto one the panel happened to hold already. A partial recall may set it on the panel's own base.
  const { hybridStartBlock } = record;

  if ((hybridBaseRecalled || (partial && values.h3HybridBaseModel !== null)) && hybridStartBlock !== null) {
    values = { ...values, h3HybridStartBlock: hybridStartBlock };
    componentsRecalled = true;
  }

  if (componentsRecalled) {
    fields.push('components');
  }

  // Promote an installed H3 checkpoint from legacy transformer metadata to model identity, even if the recorded
  // Diffusers install is gone. Drop invalid non-main overrides.
  if (values.h3TransformerModel) {
    const transformer = values.h3TransformerModel;

    if (transformer.type === 'main' && transformer.base === 'minimax-h3' && transformer.format === 'checkpoint') {
      // The recorded install (when still around) becomes the component
      // source; otherwise whatever the slot already holds is kept.
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

  const resolvedLoras = record.loras.flatMap((entry) => {
    const installed = resolveVideoModel(entry.model, models);

    return installed && isLoraModelConfig(installed) && isLoraCompatibleWithModel(installed, model)
      ? [{ isEnabled: true, model: installed as LoraModelConfig, weight: entry.weight }]
      : [];
  });

  // Reproduce the recorded LoRA set, including empty or entirely uninstalled sets; never retain unrelated panel
  // LoRAs. A partial record without `loras` leaves the panel's set alone.
  if (lorasRecalled && (resolvedLoras.length > 0 || values.loras.length > 0)) {
    const accelerator = deriveAcceleratorRecallState(model, resolvedLoras, values.steps, values);

    if (values.acceleratorEnabled && !accelerator.acceleratorEnabled) {
      // Leaving the fast path restores the model's sampling defaults, as switching it off in the panel does; a
      // value the record names stays. Otherwise the accelerator's few-step, guidance-free recipe would outlive it.
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

  // Judged against the ORIGINAL panel state: the model transition above may
  // already have cleared media the new family cannot consume, and that change
  // is still part of what this recall did.
  const hadMedia = Boolean(
    currentValues.conditioningClip ||
    currentValues.firstFrameImage ||
    currentValues.lastFrameImage ||
    currentValues.sourceVideo ||
    currentValues.references.length > 0
  );

  // All/remix recall clears current media before restoring recorded names because media determines graph family.
  // A partial recall keeps the panel's media; each slot it names displaces only its rivals when hydrated.
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

  // A partial record's explicitly empty list asks for no references, whatever other media it names; an absent one
  // leaves them alone.
  if (partial && record.referenceList === 'empty' && values.references.length > 0) {
    values = { ...values, references: [] };
    if (!fields.includes('media')) {
      fields.push('media');
    }
  }

  // Recalled alongside the source rather than with the sampling block: it is only meaningful for a
  // continuation, and it is not recoverable from anything else in the record -- the output length
  // folds the source, the generated half and the crossfade together. Snapped on the way in for the
  // same reason the settings normalizer snaps it: an off-grid value would show a count the run
  // could not use.
  const { contextFrames } = record;

  if (contextFrames !== null && model?.base === 'ltx-2') {
    values = {
      ...values,
      ltx2ExtendContextFrames: Math.max(1 + LTX2_NUM_FRAMES_STEP, snapLtx2FramesDown(contextFrames)),
    };
    fields.push('extendContext');
  }

  return fields.length > 0 ? { fields, mediaNames, values: { ...values, ...promptPatch } } : null;
};

export const getVideoRecallTitle = (kind: VideoRecallKind, t: TFunction): string =>
  t(`widgets.gallery.itemActions.recall.videoTitles.${kind}`);

export const getVideoRecallMessage = (fields: readonly VideoRecalledField[], t: TFunction): string =>
  t('widgets.gallery.itemActions.recall.appliedFields', {
    fields: fields.map((field) => t(`widgets.gallery.itemActions.recall.fields.${field}`)),
  });
