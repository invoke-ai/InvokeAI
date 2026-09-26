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
  isSupportedVideoModel,
  isValidVideoNumFrames,
  LTX2_NUM_FRAMES_STEP,
  MINIMAX_H3_HYBRID_BLOCK_RANGE,
  snapLtx2FramesDown,
  snapVideoNumFrames,
} from '@features/video';
import { SEED_MAX } from '@platform/core/seed';

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
const readKey = (metadata: unknown, key: string): unknown => {
  if (!isRecord(metadata)) {
    return undefined;
  }

  for (const candidate of [key, ...(METADATA_KEY_ALIASES[key] ?? [])]) {
    if (candidate in metadata) {
      return metadata[candidate];
    }
  }

  return undefined;
};

const getRecord = (value: unknown, key: string): Record<string, unknown> | null => {
  const child = readKey(value, key);

  return isRecord(child) ? child : null;
};

const getString = (metadata: unknown, key: string): string | null => {
  const value = readKey(metadata, key);

  return typeof value === 'string' ? value : null;
};

const getNullableString = (metadata: unknown, key: string): string | null | undefined => {
  const value = readKey(metadata, key);

  if (value === null) {
    return null;
  }

  return typeof value === 'string' ? value : undefined;
};

const getNumber = (metadata: unknown, key: string): number | null => {
  const value = readKey(metadata, key);

  return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

const getInteger = (metadata: unknown, key: string): number | null => {
  const value = getNumber(metadata, key);

  return value !== null && Number.isInteger(value) ? value : null;
};

const getSeed = (metadata: unknown): number | null => {
  const seed = getInteger(metadata, 'seed');

  return seed !== null && seed >= 0 && seed <= SEED_MAX ? seed : null;
};

export const isVideoGenerationMetadata = (metadata: unknown): boolean => {
  const mode = getString(metadata, 'generation_mode');

  return mode !== null && VIDEO_GENERATION_MODE_IDS.has(mode);
};

/** A recorded `ModelIdentifier` — `{ key, hash, name, base, type }` — with every field optional on read. */
interface RecordedModelRef {
  key: string | null;
  hash: string | null;
  name: string | null;
  base: string | null;
  type: string | null;
}

const toRecordedModelRef = (value: unknown): RecordedModelRef | null => {
  if (!isRecord(value)) {
    return null;
  }

  const field = (name: string) => (typeof value[name] === 'string' ? (value[name] as string) : null);
  const ref: RecordedModelRef = {
    base: field('base'),
    hash: field('hash'),
    key: field('key'),
    name: field('name'),
    type: field('type'),
  };

  // Only a ref that at least one rung of `resolveRecordedModel` can act on counts as recorded;
  // anything less would advertise a recall that can never resolve.
  return ref.key || ref.hash || (ref.name && ref.base && ref.type) ? ref : null;
};

const getMetadataModelRef = (metadata: unknown, key: string): RecordedModelRef | null =>
  toRecordedModelRef(getRecord(metadata, key));

/**
 * Resolve install-local keys, then portable content hashes, then name/base/type. Only the final fallback may
 * select different weights.
 */
const resolveRecordedModel = <T extends ModelConfig>(ref: RecordedModelRef | null, models: readonly T[]): T | null => {
  if (!ref) {
    return null;
  }

  const byKey = ref.key ? models.find((model) => model.key === ref.key) : undefined;
  if (byKey) {
    return byKey;
  }

  const byHash = ref.hash ? models.find((model) => model.hash === ref.hash) : undefined;
  if (byHash) {
    return byHash;
  }

  if (ref.name && ref.base && ref.type) {
    return (
      models.find((model) => model.name === ref.name && model.base === ref.base && model.type === ref.type) ?? null
    );
  }

  return null;
};

const getSupportedVideoMetadataModel = (metadata: unknown, models: readonly ModelConfig[]): MainModelConfig | null => {
  const installed = resolveRecordedModel(getMetadataModelRef(metadata, 'model'), models);

  return installed && isSupportedVideoModel(installed) ? (installed as MainModelConfig) : null;
};

const getMetadataPrompts = (
  metadata: unknown
): { positivePrompt: string | null; negativePrompt: string | null | undefined } => ({
  negativePrompt: getNullableString(metadata, 'negative_prompt'),
  positivePrompt: getString(metadata, 'positive_prompt'),
});

const hasPrompt = (metadata: unknown): boolean => {
  const { negativePrompt, positivePrompt } = getMetadataPrompts(metadata);

  return positivePrompt !== null || typeof negativePrompt === 'string';
};

const getImageName = (metadata: unknown, key: string): string | null => {
  const field = getRecord(metadata, key);

  return typeof field?.image_name === 'string' ? field.image_name : null;
};

/** The whole-modality conditioning clip, from the LTX-2 extras the graph records. */
const getRecallableConditioningClip = (metadata: unknown): VideoRecallConditioningClip | null => {
  const field = getRecord(metadata, 'ltx2_conditioning_video');
  const role = isRecord(metadata) ? metadata.ltx2_conditioning_role : undefined;

  if (typeof field?.video_name !== 'string' || (role !== 'audio' && role !== 'video')) {
    return null;
  }

  return { name: field.video_name, role };
};

const getSourceVideoName = (metadata: unknown): string | null => {
  const field = getRecord(metadata, 'source_video');

  return typeof field?.video_name === 'string' ? field.video_name : null;
};

/**
 * In extend mode, first_frame_image was extracted from the source clip; do not restore it as user input or
 * conflict with initial video.
 */
export const getRecallableMediaNames = (
  metadata: unknown
): { firstFrameName: string | null; lastFrameName: string | null; sourceVideoName: string | null } => {
  const sourceVideoName = getSourceVideoName(metadata);

  return {
    firstFrameName: sourceVideoName ? null : getImageName(metadata, 'first_frame_image'),
    lastFrameName: getImageName(metadata, 'last_frame_image'),
    sourceVideoName,
  };
};

const getMetadataLoras = (metadata: unknown): { model: RecordedModelRef; weight: number }[] => {
  if (!isRecord(metadata) || !Array.isArray(metadata.loras)) {
    return [];
  }

  return metadata.loras.flatMap((entry) => {
    const model = isRecord(entry) ? toRecordedModelRef(entry.model) : null;

    if (!model) {
      return [];
    }

    const weight = typeof entry.weight === 'number' && Number.isFinite(entry.weight) ? entry.weight : 1;

    return [{ model, weight }];
  });
};

/** Metadata component slot → widget-values key, with the recorded model resolved against the catalog. */
const VIDEO_COMPONENT_METADATA_KEYS = [
  ['vae', 'vae'],
  ['wan_t5_encoder_model', 'wanT5EncoderModel'],
  ['wan_component_source', 'componentSourceModel'],
  ['wan_transformer_low_noise', 'wanLowNoiseModel'],
  ['minimax_h3_transformer_model', 'h3TransformerModel'],
  ['minimax_h3_component_source', 'componentSourceModel'],
  ['minimax_h3_text_encoder_model', 'h3TextEncoderModel'],
  ['minimax_h3_hybrid_base_model', 'h3HybridBaseModel'],
  ['ltx2_component_source', 'componentSourceModel'],
  ['ltx2_text_encoder_model', 'ltx2TextEncoderModel'],
] as const;

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

export const getVideoRecallCapabilities = (metadata: unknown): VideoRecallCapabilities => {
  if (!isVideoGenerationMetadata(metadata)) {
    return EMPTY_VIDEO_RECALL_CAPABILITIES;
  }

  const prompts = hasPrompt(metadata);
  const seed = getSeed(metadata) !== null;
  const media = getRecallableMediaNames(metadata);
  const hasNonSeed =
    prompts ||
    getMetadataModelRef(metadata, 'model') !== null ||
    getInteger(metadata, 'num_frames') !== null ||
    getInteger(metadata, 'steps') !== null ||
    getNumber(metadata, 'cfg_scale') !== null ||
    getMetadataLoras(metadata).length > 0 ||
    media.firstFrameName !== null ||
    media.lastFrameName !== null ||
    media.sourceVideoName !== null;

  return {
    all: hasNonSeed || seed,
    prompts,
    remix: hasNonSeed,
    seed,
  };
};

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
  /** The LTX-2 whole-modality conditioning clip, which excludes every other slot below. */
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

/** Tolerant parser for the `minimax_h3_references` metadata extra: skips malformed entries, keeps order. */
const getRecallableReferences = (metadata: unknown): VideoRecallReferenceName[] => {
  if (!isRecord(metadata) || !Array.isArray(metadata.minimax_h3_references)) {
    return [];
  }

  // Cap at the request maximum (3 videos + 9 images): corrupt metadata must not drive an
  // unbounded chain of gallery resolves in the executor.
  return metadata.minimax_h3_references.slice(0, 12).flatMap((entry): VideoRecallReferenceName[] => {
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
  if (requireGenerationMode ? !isVideoGenerationMetadata(metadata) : !isRecord(metadata)) {
    return null;
  }

  const fields: VideoRecalledField[] = [];
  let values: VideoWidgetValues = { ...currentValues };
  // Hold prompts outside values until return so model-default transitions cannot overwrite them.
  let promptPatch: Partial<VideoWidgetValues> | null = null;
  const mediaNames: VideoRecallMediaNames = {
    conditioningClip: null,
    firstFrameName: null,
    lastFrameName: null,
    references: [],
    sourceVideoName: null,
    sourceVideoTrim: null,
  };

  if (kind === 'all' || kind === 'prompts' || kind === 'remix') {
    const { negativePrompt, positivePrompt } = getMetadataPrompts(metadata);

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
    const seed = getSeed(metadata);

    if (seed !== null) {
      values = { ...values, seed, seedMode: 'fixed' };
      fields.push('seed');
    }
  }

  if (kind === 'seed') {
    return fields.length > 0 ? { fields, mediaNames, values: { ...values, ...promptPatch } } : null;
  }

  // all / remix from here on.
  const recalledModel = getSupportedVideoMetadataModel(metadata, models);
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

  const numFrames = getInteger(metadata, 'num_frames');

  if (numFrames !== null && numFrames > 0) {
    const snapped = isValidVideoNumFrames(model, numFrames) ? numFrames : snapVideoNumFrames(model, numFrames);

    if (snapped !== values.numFrames) {
      values = { ...values, numFrames: snapped };
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
  const lorasRecalled = !partial || (isRecord(metadata) && Array.isArray(metadata.loras));
  const policy = getVideoModelPolicy(model, lorasRecalled ? { ...values, acceleratorEnabled: false } : values);
  const steps = getInteger(metadata, 'steps');

  // A fixed-schedule checkpoint ignores whatever step count reaches it, so recalling one would
  // leave a disabled control showing a number the run will not use — and re-record it next time.
  if (policy.ui.stepsEditable && steps !== null && steps >= 1) {
    values = { ...values, steps };
    fields.push('steps');
  }

  const cfgScale = getNumber(metadata, 'cfg_scale');
  const cfgScaleLowNoise = getNumber(metadata, 'wan_guidance_scale_low_noise');

  if (policy.ui.cfgVisible && cfgScale !== null && cfgScale >= 1) {
    values = {
      ...values,
      cfgScale,
      // The graph records the key only when the setting was non-null; absence
      // means "reuse the primary CFG", so restore that exact semantics.
      ...(policy.ui.cfgLowNoiseVisible
        ? { cfgScaleLowNoise: cfgScaleLowNoise !== null && cfgScaleLowNoise >= 1 ? cfgScaleLowNoise : null }
        : {}),
    };
    fields.push('cfg');
  }

  // The per-modality scales ride with CFG: they are the same run's guidance,
  // and a family that does not offer a control must not be handed a number.
  const guidanceRecall = [
    { floor: 1, key: 'audioCfgScale', metadataKey: 'ltx2_audio_cfg_scale', visible: policy.ui.audioCfgVisible },
    { floor: 0, key: 'stgScale', metadataKey: 'ltx2_stg_scale', visible: policy.ui.stgVisible },
    { floor: 1, key: 'modalityScale', metadataKey: 'ltx2_modality_scale', visible: policy.ui.modalityVisible },
  ] as const;

  for (const { floor, key, metadataKey, visible } of guidanceRecall) {
    const recalled = getNumber(metadata, metadataKey);

    if (visible && recalled !== null && recalled >= floor) {
      values = { ...values, [key]: recalled };

      if (!fields.includes('cfg')) {
        fields.push('cfg');
      }
    }
  }

  const fps = getInteger(metadata, 'fps');

  // Wan records delivered fps; extend compilation still inherits clip fps, and H3 fixes it at 24.
  if (policy.fps.editable && fps !== null && fps >= 1 && fps <= 120) {
    if (fps !== values.fps) {
      values = { ...values, fps };
    }
    fields.push('fps');
  }

  const sizeRecall = getVideoSizeRecall(model, getInteger(metadata, 'width'), getInteger(metadata, 'height'));

  if (sizeRecall) {
    values = { ...values, ...sizeRecall };
    fields.push('size');
  }

  // Recall components before deriving acceleration so H3 task detection sees the recorded transformer.
  let componentsRecalled = false;
  let hybridBaseRecalled = false;

  for (const [metadataKey, valuesKey] of VIDEO_COMPONENT_METADATA_KEYS) {
    const installed = resolveRecordedModel(getMetadataModelRef(metadata, metadataKey), models);

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
  const hybridStartBlock = getInteger(metadata, 'minimax_h3_hybrid_start_block');

  if (
    (hybridBaseRecalled || (partial && values.h3HybridBaseModel !== null)) &&
    hybridStartBlock !== null &&
    hybridStartBlock >= MINIMAX_H3_HYBRID_BLOCK_RANGE.min &&
    hybridStartBlock <= MINIMAX_H3_HYBRID_BLOCK_RANGE.max
  ) {
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

  const recordedLoras = getMetadataLoras(metadata);
  const resolvedLoras = recordedLoras.flatMap((entry) => {
    const installed = resolveRecordedModel(entry.model, models);

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
      const recorded = (key: string) => getNumber(metadata, key) !== null;

      values = {
        ...values,
        ...(recorded('steps') ? {} : { steps: defaults.steps }),
        ...(recorded('cfg_scale') ? {} : { cfgScale: defaults.cfgScale, cfgScaleLowNoise: defaults.cfgScaleLowNoise }),
        ...(recorded('ltx2_audio_cfg_scale') ? {} : { audioCfgScale: defaults.audioCfgScale }),
        ...(recorded('ltx2_stg_scale') ? {} : { stgScale: defaults.stgScale }),
        ...(recorded('ltx2_modality_scale') ? {} : { modalityScale: defaults.modalityScale }),
      };
    }
    values = { ...values, loras: resolvedLoras, ...accelerator };
    fields.push('loras');
  }

  const media = getRecallableMediaNames(metadata);
  const references = getRecallableReferences(metadata);
  const conditioningClip = getRecallableConditioningClip(metadata);
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

  if (conditioningClip) {
    // First, and alone: the clip holds a whole modality clean, so no other slot could have been
    // filled on the run being recalled.
    mediaNames.conditioningClip = conditioningClip;
    fields.push('media');
  } else if (references.length > 0) {
    // References replace the frame slots, but a recorded source video rides
    // ALONGSIDE them: Ref2VA reference-extend appends the new clip to it.
    mediaNames.references = references;
    if (media.sourceVideoName) {
      const startFrame = getInteger(metadata, 'source_video_start_frame');
      const endFrame = getInteger(metadata, 'source_video_end_frame');

      mediaNames.sourceVideoName = media.sourceVideoName;
      mediaNames.sourceVideoTrim = startFrame !== null && endFrame !== null ? { endFrame, startFrame } : null;
    }
    fields.push('media');
  } else if (media.firstFrameName || media.lastFrameName || media.sourceVideoName) {
    mediaNames.firstFrameName = media.firstFrameName;
    mediaNames.lastFrameName = media.lastFrameName;
    mediaNames.sourceVideoName = media.sourceVideoName;
    if (media.sourceVideoName) {
      const startFrame = getInteger(metadata, 'source_video_start_frame');
      const endFrame = getInteger(metadata, 'source_video_end_frame');

      mediaNames.sourceVideoTrim = startFrame !== null && endFrame !== null ? { endFrame, startFrame } : null;
    }
    fields.push('media');
  } else if (hadMedia && !partial) {
    fields.push('media');
  }

  // A partial record's explicitly empty list asks for no references, whatever other media it names; an absent one
  // leaves them alone.
  if (
    partial &&
    isRecord(metadata) &&
    Array.isArray(metadata.minimax_h3_references) &&
    metadata.minimax_h3_references.length === 0 &&
    values.references.length > 0
  ) {
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
  const contextFrames = getInteger(metadata, 'ltx2_context_frames');

  if (contextFrames !== null && model?.base === 'ltx-2') {
    values = {
      ...values,
      ltx2ExtendContextFrames: Math.max(1 + LTX2_NUM_FRAMES_STEP, snapLtx2FramesDown(contextFrames)),
    };
    fields.push('extendContext');
  }

  return fields.length > 0 ? { fields, mediaNames, values: { ...values, ...promptPatch } } : null;
};

const VIDEO_RECALL_TITLES: Record<VideoRecallKind, string> = {
  all: 'Recalled video data',
  prompts: 'Recalled prompts',
  remix: 'Remixed video',
  seed: 'Recalled seed',
};

export const getVideoRecallTitle = (kind: VideoRecallKind): string => VIDEO_RECALL_TITLES[kind];

const VIDEO_FIELD_LABELS: Record<VideoRecalledField, string> = {
  cfg: 'CFG',
  steps: 'steps',
  components: 'components',
  extendContext: 'context frames',
  fps: 'FPS',
  frames: 'frames',
  loras: 'concepts',
  media: 'conditioning media',
  model: 'model',
  prompts: 'prompts',
  seed: 'seed',
  size: 'size',
};

export const getVideoRecallMessage = (fields: readonly VideoRecalledField[]): string =>
  `Applied ${fields.map((field) => VIDEO_FIELD_LABELS[field]).join(', ')}.`;
