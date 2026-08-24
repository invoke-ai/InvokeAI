import type {
  GenerateLora,
  GenerationModelCatalogItem as ModelConfig,
  LoraModelConfig,
  MainModelConfig,
} from '@features/generation/contracts';
import type { ProjectPromptDraftPatch } from '@features/generation/settings';
import type { VideoAspectRatioId, VideoTargetResolution, VideoWidgetValues } from '@features/video';

import { isLoraCompatibleWithModel, isLoraModelConfig, SEED_MAX } from '@features/generation/settings';
import {
  findMiniMaxH3TurboLora,
  findWanLightningLoraPair,
  getVideoAspectRatioOptions,
  getVideoDimensions,
  getVideoModelPolicy,
  getVideoModelSelectionResult,
  getVideoTargetResolutionOptions,
  isSupportedVideoModel,
  isValidVideoNumFrames,
  snapVideoNumFrames,
} from '@features/video';

/**
 * Pure mapping from a video's recorded `core_metadata` to a Video-panel
 * patch — the sibling of `imageRecall.ts`. Model resolution is by KEY against
 * the installed catalog; an uninstalled model skips the model field and
 * everything downstream validates against the current model instead. The
 * conditioning mode is never recalled directly: the panel derives it from
 * which media fields are filled, so recall restores the media and lets the
 * mode fall out.
 */

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
  | 'media';

export interface VideoRecallResult {
  fields: VideoRecalledField[];
  /** Prompt fields live in the shared project draft, not the widget values. */
  promptPatch: ProjectPromptDraftPatch | null;
  values: VideoWidgetValues;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const getRecord = (value: unknown, key: string): Record<string, unknown> | null => {
  if (!isRecord(value)) {
    return null;
  }

  const child = value[key];

  return isRecord(child) ? child : null;
};

const getString = (metadata: unknown, key: string): string | null => {
  if (!isRecord(metadata)) {
    return null;
  }

  const value = metadata[key];

  return typeof value === 'string' ? value : null;
};

const getNullableString = (metadata: unknown, key: string): string | null | undefined => {
  if (!isRecord(metadata)) {
    return undefined;
  }

  const value = metadata[key];

  if (value === null) {
    return null;
  }

  return typeof value === 'string' ? value : undefined;
};

const getNumber = (metadata: unknown, key: string): number | null => {
  if (!isRecord(metadata)) {
    return null;
  }

  const value = metadata[key];

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

const getMetadataModelKey = (metadata: unknown, key: string): string | null => {
  const model = getRecord(metadata, key);

  return typeof model?.key === 'string' ? model.key : null;
};

const getSupportedVideoMetadataModel = (metadata: unknown, models: readonly ModelConfig[]): MainModelConfig | null => {
  const modelKey = getMetadataModelKey(metadata, 'model');

  if (!modelKey) {
    return null;
  }

  const installed = models.find((model) => model.key === modelKey);

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

const getSourceVideoName = (metadata: unknown): string | null => {
  const field = getRecord(metadata, 'source_video');

  return typeof field?.video_name === 'string' ? field.video_name : null;
};

/**
 * The keyframe names the recall should restore, per the recorded mode. In
 * extend mode `first_frame_image` is the frame the graph EXTRACTED from the
 * source clip — derived conditioning, not user input — so restoring it would
 * both misrepresent the session and collide with the panel's first-frame ⊻
 * initial-video exclusion.
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

const getMetadataLoras = (metadata: unknown): { key: string; weight: number }[] => {
  if (!isRecord(metadata) || !Array.isArray(metadata.loras)) {
    return [];
  }

  return metadata.loras.flatMap((entry) => {
    if (!isRecord(entry) || !isRecord(entry.model) || typeof entry.model.key !== 'string') {
      return [];
    }

    const weight = typeof entry.weight === 'number' && Number.isFinite(entry.weight) ? entry.weight : 1;

    return [{ key: entry.model.key, weight }];
  });
};

/** Metadata component slot → widget-values key, with the recorded value's key resolved against the catalog. */
const VIDEO_COMPONENT_METADATA_KEYS = [
  ['vae', 'vae'],
  ['wan_t5_encoder', 'wanT5EncoderModel'],
  ['wan_component_source', 'componentSourceModel'],
  ['transformer_low_noise', 'wanLowNoiseModel'],
  ['minimax_h3_transformer_model', 'h3TransformerModel'],
  ['minimax_h3_text_encoder_model', 'h3TextEncoderModel'],
] as const;

/**
 * The recorded W×H inverted back into the panel's aspect-ratio preset +
 * target-resolution pair — exact matches only. A near-miss would silently
 * recall a preset the run never used: a non-matching size means the dims came
 * from conditioning media, and recalling that media locks the size anyway.
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
 * Whether the recalled LoRA list is exactly what the accelerator toggle would
 * install for this model — if so the flag (and its recorded keys) come back
 * on, so the panel shows the same fast-path state that produced the video.
 */
export const deriveAcceleratorRecallState = (
  model: MainModelConfig,
  models: readonly ModelConfig[],
  loras: readonly GenerateLora[],
  steps: number,
  settings: VideoWidgetValues
): Pick<VideoWidgetValues, 'acceleratorEnabled' | 'acceleratorLoraKeys'> => {
  const accelerator = getVideoModelPolicy(model, settings).ui.accelerator;

  if (!accelerator || steps !== accelerator.steps) {
    return { acceleratorEnabled: false, acceleratorLoraKeys: [] };
  }

  const expected =
    model.base === 'minimax-h3'
      ? [findMiniMaxH3TurboLora(models)].flatMap((lora) => (lora ? [lora.key] : []))
      : (() => {
          const pair = findWanLightningLoraPair(models, model.variant);

          return pair ? [pair.high.key, pair.low.key] : [];
        })();
  const present = expected.length > 0 && expected.every((key) => loras.some((lora) => lora.model.key === key));

  return present
    ? { acceleratorEnabled: true, acceleratorLoraKeys: expected }
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
    getMetadataModelKey(metadata, 'model') !== null ||
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
export interface VideoRecallMediaNames {
  firstFrameName: string | null;
  lastFrameName: string | null;
  sourceVideoName: string | null;
  /**
   * The recorded trim bounds (metadata extras), so the executor can restore
   * the clip's actual trim instead of `createVideoSourceClip`'s default —
   * without them a recalled extension would start from the wrong frame.
   */
  sourceVideoTrim: { endFrame: number; startFrame: number } | null;
}

export const buildVideoRecallSettings = ({
  currentValues,
  kind,
  metadata,
  models,
}: {
  currentValues: VideoWidgetValues;
  kind: VideoRecallKind;
  metadata: unknown;
  models: readonly ModelConfig[];
}): (VideoRecallResult & { mediaNames: VideoRecallMediaNames }) | null => {
  if (!isVideoGenerationMetadata(metadata)) {
    return null;
  }

  const fields: VideoRecalledField[] = [];
  let values: VideoWidgetValues = { ...currentValues };
  let promptPatch: ProjectPromptDraftPatch | null = null;
  const mediaNames: VideoRecallMediaNames = {
    firstFrameName: null,
    lastFrameName: null,
    sourceVideoName: null,
    sourceVideoTrim: null,
  };

  if (kind === 'all' || kind === 'prompts' || kind === 'remix') {
    const { negativePrompt, positivePrompt } = getMetadataPrompts(metadata);

    if (positivePrompt !== null || negativePrompt !== undefined) {
      // A disabled negative submits as '' — an empty recorded negative is
      // indistinguishable from disabled, so it must not flip the toggle on
      // (or erase a saved-but-disabled draft's enabled state).
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
    return fields.length > 0 ? { fields, mediaNames, promptPatch, values } : null;
  }

  if (kind === 'all' || kind === 'seed') {
    const seed = getSeed(metadata);

    if (seed !== null) {
      values = { ...values, seed, shouldRandomizeSeed: false };
      fields.push('seed');
    }
  }

  if (kind === 'seed') {
    return fields.length > 0 ? { fields, mediaNames, promptPatch, values } : null;
  }

  // all / remix from here on.
  const recalledModel = getSupportedVideoMetadataModel(metadata, models);
  const model = recalledModel ?? values.model;

  if (recalledModel && recalledModel.key !== values.model?.key) {
    // The canonical family transition first, so frames/fps/resolution snap to
    // the recalled model before its recorded values land on top.
    values = {
      ...getVideoModelSelectionResult({ currentSettings: values, model: recalledModel, models }).settings,
      model: recalledModel,
    };
    fields.push('model');
  } else if (recalledModel) {
    fields.push('model');
  }

  if (!model) {
    // Without any model, nothing below can validate; prompts/seed may still
    // have been recalled.
    return fields.length > 0 ? { fields, mediaNames, promptPatch, values } : null;
  }

  const numFrames = getInteger(metadata, 'num_frames');

  if (numFrames !== null && numFrames > 0) {
    const snapped = isValidVideoNumFrames(model, numFrames) ? numFrames : snapVideoNumFrames(model, numFrames);

    if (snapped !== values.numFrames) {
      values = { ...values, numFrames: snapped };
    }
    fields.push('frames');
  }

  const steps = getInteger(metadata, 'steps');

  if (steps !== null && steps >= 1) {
    values = { ...values, steps };
    fields.push('steps');
  }

  const cfgScale = getNumber(metadata, 'cfg_scale');
  const cfgScaleLowNoise = getNumber(metadata, 'guidance_scale_low_noise');
  const policy = getVideoModelPolicy(model, values);

  if (policy.ui.cfgVisible && cfgScale !== null && cfgScale >= 1) {
    values = {
      ...values,
      cfgScale,
      // The graph records the key only when the setting was non-null; absence
      // means "reuse the primary CFG", so restore that exact semantics.
      ...(policy.ui.cfgLowNoiseVisible
        ? { cfgScaleLowNoise: cfgScaleLowNoise !== null && cfgScaleLowNoise >= 0 ? cfgScaleLowNoise : null }
        : {}),
    };
    fields.push('cfg');
  }

  const fps = getInteger(metadata, 'fps');

  // Wan records the delivered frame rate (H3 is fixed at 24 and records
  // none). Recalling it into an extend-mode panel is harmless: the fps field
  // is display-only there and the compiled graph re-inherits the clip's rate.
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

  const recordedLoras = getMetadataLoras(metadata);
  const resolvedLoras = recordedLoras.flatMap((entry) => {
    const installed = models.find((candidate) => candidate.key === entry.key);

    return installed && isLoraModelConfig(installed) && isLoraCompatibleWithModel(installed, model)
      ? [{ isEnabled: true, model: installed as LoraModelConfig, weight: entry.weight }]
      : [];
  });

  // The recall reproduces the recorded LoRA set: empty when the video ran
  // without LoRAs, and also when the recorded ones are no longer installed —
  // keeping the panel's current LoRAs would misrepresent the run either way.
  if (resolvedLoras.length > 0 || values.loras.length > 0) {
    values = {
      ...values,
      loras: resolvedLoras,
      ...deriveAcceleratorRecallState(model, models, resolvedLoras, values.steps, values),
    };
    fields.push('loras');
  }

  let componentsRecalled = false;

  for (const [metadataKey, valuesKey] of VIDEO_COMPONENT_METADATA_KEYS) {
    const recordedKey = getMetadataModelKey(metadata, metadataKey);

    if (!recordedKey) {
      continue;
    }

    const installed = models.find((candidate) => candidate.key === recordedKey);

    if (installed) {
      values = { ...values, [valuesKey]: installed };
      componentsRecalled = true;
    }
  }

  if (componentsRecalled) {
    fields.push('components');
  }

  const media = getRecallableMediaNames(metadata);
  // Judged against the ORIGINAL panel state: the model transition above may
  // already have cleared media the new family cannot consume, and that change
  // is still part of what this recall did.
  const hadMedia = Boolean(currentValues.firstFrameImage || currentValues.lastFrameImage || currentValues.sourceVideo);

  // Media selects the graph family (t2v vs i2v vs extend), so an all/remix
  // recall must reproduce the recorded media EXACTLY: whatever the panel held
  // is cleared, and the executor re-hydrates the recorded names on top.
  if (hadMedia) {
    values = { ...values, firstFrameImage: null, lastFrameImage: null, sourceVideo: null };
  }

  if (media.firstFrameName || media.lastFrameName || media.sourceVideoName) {
    mediaNames.firstFrameName = media.firstFrameName;
    mediaNames.lastFrameName = media.lastFrameName;
    mediaNames.sourceVideoName = media.sourceVideoName;
    if (media.sourceVideoName) {
      const startFrame = getInteger(metadata, 'source_video_start_frame');
      const endFrame = getInteger(metadata, 'source_video_end_frame');

      mediaNames.sourceVideoTrim = startFrame !== null && endFrame !== null ? { endFrame, startFrame } : null;
    }
    fields.push('media');
  } else if (hadMedia) {
    fields.push('media');
  }

  return fields.length > 0 ? { fields, mediaNames, promptPatch, values } : null;
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
