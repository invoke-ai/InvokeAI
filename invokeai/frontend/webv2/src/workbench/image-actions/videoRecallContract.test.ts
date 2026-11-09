/**
 * Every key a video graph writes into its metadata node is accounted for here: recall restores it, or it is
 * deliberately inert. The video twin of `imageRecallContract.test.ts`.
 *
 * The cases are the registry-driven matrix `videoGraphCoverage.test.ts` compiles -- every base, variant, mode, model
 * shape and multi-stage preset, with and without auto duration -- each run with every recallable value tuned away
 * from its default and every component slot filled, so each key a builder can write is written. A key with no entry
 * in `VIDEO_RECALL_CONTRACT` fails G1, so a new parameter cannot reach a record untested:
 *
 * - `restores`: a full recall puts the value back (G2), and only through this key: recalling the record without it
 *   must land somewhere else, or the round trip proves nothing. `when` is false on runs recall deliberately leaves
 *   the value alone on, and G3 then holds the key inert there.
 * - `media`: recall names the medium for the executor to hydrate from the gallery (G2, against `mediaNames`).
 * - `ignored`: recall applies nothing from it; recalling the record without it changes nothing (G3).
 * - `consumed`: read to interpret the record rather than restored, so it cannot be ablated.
 *
 * A value only the run produces arrives over an edge; it is recorded as `<runtime:…>` and classified in
 * `VIDEO_RUNTIME_CONTRACT` by key and source node type.
 */

import type {
  GenerateLora,
  GenerationModelCatalogItem as ModelConfig,
  LoraModelConfig,
  MainModelConfig,
} from '@features/generation/contracts';
import type {
  VideoComponentSlotPolicy,
  VideoGenerationMode,
  VideoModelPolicy,
  VideoReferenceItem,
  VideoSettings,
  VideoSourceClip,
  VideoWidgetValues,
} from '@features/video';
import type { Case } from '@features/video/core/videoGraphCoverage.testing';
import type { TFunction } from 'i18next';

import {
  compileVideoGraph,
  createDefaultVideoWidgetValues,
  getDefaultVideoSettings,
  getVideoComponentSectionPolicy,
  getVideoDimensions,
  getVideoModelPolicy,
  getVideoValidationReasons,
  resolveVideoMode,
} from '@features/video';
import {
  AUTO_DURATION_CASES,
  candidatesForSlot,
  cases,
  createModel,
  DURATION_HEAD,
} from '@features/video/core/videoGraphCoverage.testing';
import i18n from 'i18next';
import { beforeAll, describe, expect, it } from 'vitest';

import type {
  VideoRecalledField,
  VideoRecallKind,
  VideoRecallMediaNames,
  VideoRecallReferenceName,
} from './videoRecall';

import { recordOf, runtimeSourceOf } from './recallContract.testing';
import {
  buildVideoRecallSettings,
  getVideoRecallCapabilities,
  getVideoRecallMessage,
  getVideoRecallTitle,
  getVideoSizeRecalls,
  parseVideoRecallRecord,
  settleMediaSizedRecall,
} from './videoRecall';

/** What a contract entry may consult: the recorded run, as tuned, and what its graph wrote. */
interface RecordedRun {
  model: MainModelConfig;
  /** The tuned run's own policy, with its accelerator off as every tuned run has it. */
  policy: VideoModelPolicy;
  tuned: VideoWidgetValues;
  written: Record<string, unknown>;
}

type WidgetValueKeys = readonly (keyof VideoWidgetValues)[];

interface RestoresEntry {
  /** The values it puts back; per run where what it restores depends on the run. */
  restores: WidgetValueKeys | ((run: RecordedRun) => WidgetValueKeys);
  /** The field the recall notice names for it. */
  reports: VideoRecalledField;
  /** The recall verbs that apply it; all and remix when absent. */
  kinds?: readonly VideoRecallKind[];
  /** False for a run recall deliberately leaves the value alone on; G3 then holds the key inert. */
  when?: (run: RecordedRun) => boolean;
  /** The restored value where it is the run's rather than the panel's (a rate or length taken from media). */
  expected?: (run: RecordedRun) => Partial<VideoWidgetValues>;
  /** What a full recall of a record without the key leaves on a panel holding `panel`: the run did not use one. */
  absent?: { panel: Partial<VideoWidgetValues>; restored: Partial<VideoWidgetValues> };
}

interface MediaEntry {
  media: (run: RecordedRun) => Partial<VideoRecallMediaNames>;
}

type ReasonEntry = { consumed: string } | { ignored: string };
type Entry = RestoresEntry | MediaEntry | ReasonEntry;

const PANEL_LORA: LoraModelConfig = { base: 'wan', key: 'panel-lora', name: 'Panel style LoRA', type: 'lora' };
const PANEL_HYBRID_BASE: MainModelConfig = {
  base: 'minimax-h3',
  format: 'checkpoint',
  key: 'panel-h3-fl2va',
  name: 'Panel H3 FL2VA transformer',
  type: 'main',
  variant: 'fl2va',
};

const PROMPT_KINDS = ['all', 'remix', 'prompts'] as const;

// A canvas conditioning media sized keeps their shape, so its aspect ratio stays theirs and only the target resolution
// that sized it comes back.
const SIZE: RestoresEntry = {
  kinds: ['all', 'remix', 'size'],
  reports: 'size',
  restores: ({ model, tuned }) =>
    getVideoDimensions(model, tuned)?.source === 'aspect-ratio'
      ? ['aspectRatioId', 'targetResolution']
      : ['targetResolution'],
};

const SOURCE_TRIM: MediaEntry = {
  media: ({ tuned }) => ({
    sourceVideoTrim: { endFrame: tuned.sourceVideo!.endFrame, startFrame: tuned.sourceVideo!.startFrame },
  }),
};

const CONDITIONING_CLIP: MediaEntry = {
  media: ({ tuned }) => ({
    conditioningClip: { name: tuned.conditioningClip!.clip.video_name, role: tuned.conditioningClip!.role },
  }),
};

const toReferenceName = (reference: VideoReferenceItem): VideoRecallReferenceName =>
  reference.kind === 'image'
    ? { detail: reference.detail, kind: 'image', name: reference.image.image_name }
    : {
        conditioning: reference.conditioning,
        kind: 'video',
        name: reference.clip.video_name,
        trim: { endFrame: reference.clip.endFrame, startFrame: reference.clip.startFrame },
      };

const VIDEO_RECALL_CONTRACT: Readonly<Record<string, Entry>> = {
  cfg_scale: { reports: 'cfg', restores: ['cfgScale'], when: ({ policy }) => policy.ui.cfgVisible },
  first_frame_image: { media: ({ tuned }) => ({ firstFrameName: tuned.firstFrameImage!.image_name }) },
  // The delivered rate: an extension inherits its source's, and a picture-role clip its own.
  fps: {
    expected: ({ written }) => ({ fps: written.fps as number }),
    reports: 'fps',
    restores: ['fps'],
    when: ({ policy }) => policy.fps.editable,
  },
  generation_mode: {
    consumed:
      'The record discriminator: only a mode the video graphs stamp makes the record recallable at all, and only an ' +
      'H3 one lets a legacy duration count as frames. Without it nothing is recalled, so it cannot be ablated.',
  },
  height: SIZE,
  last_frame_image: { media: ({ tuned }) => ({ lastFrameName: tuned.lastFrameImage!.image_name }) },
  loras: {
    absent: { panel: { loras: [{ isEnabled: true, model: PANEL_LORA, weight: 1 }] }, restored: { loras: [] } },
    reports: 'loras',
    restores: ['loras'],
  },
  ltx2_audio_cfg_scale: { reports: 'cfg', restores: ['audioCfgScale'] },
  ltx2_base_height: {
    ignored: "Informational: a two-stage run's base pass is half its canvas, and recall recovers the preset from it.",
  },
  ltx2_base_width: {
    ignored: "Informational: a two-stage run's base pass is half its canvas, and recall recovers the preset from it.",
  },
  ltx2_component_source: { reports: 'components', restores: ['componentSourceModel'] },
  ltx2_conditioning_role: CONDITIONING_CLIP,
  ltx2_conditioning_video: CONDITIONING_CLIP,
  ltx2_context_frames: { reports: 'extendContext', restores: ['ltx2ExtendContextFrames'] },
  ltx2_modality_scale: { reports: 'cfg', restores: ['modalityScale'] },
  ltx2_stg_scale: { reports: 'cfg', restores: ['stgScale'] },
  ltx2_text_encoder_model: { reports: 'components', restores: ['ltx2TextEncoderModel'] },
  ltx2_two_stage: { ignored: 'Informational: the preset recall recovers from the canvas states its stage count.' },
  minimax_h3_component_source: { reports: 'components', restores: ['componentSourceModel'] },
  minimax_h3_hybrid_base_model: {
    absent: { panel: { h3HybridBaseModel: PANEL_HYBRID_BASE }, restored: { h3HybridBaseModel: null } },
    reports: 'components',
    restores: ['h3HybridBaseModel'],
  },
  minimax_h3_hybrid_start_block: { reports: 'components', restores: ['h3HybridStartBlock'] },
  minimax_h3_references: { media: ({ tuned }) => ({ references: tuned.references.map(toReferenceName) }) },
  minimax_h3_text_encoder_model: { reports: 'components', restores: ['h3TextEncoderModel'] },
  model: { reports: 'model', restores: ['model', 'modelKey'] },
  negative_prompt: { kinds: PROMPT_KINDS, reports: 'prompts', restores: ['negativePrompt', 'negativePromptEnabled'] },
  // The length that ran: a conditioning clip decides it rather than the panel's stored count.
  num_frames: {
    expected: ({ written }) => ({ numFrames: written.num_frames as number }),
    reports: 'frames',
    restores: ['numFrames'],
  },
  positive_prompt: { kinds: PROMPT_KINDS, reports: 'prompts', restores: ['positivePrompt'] },
  seed: { kinds: ['all', 'seed'], reports: 'seed', restores: ['seed', 'seedMode'] },
  source_video: { media: ({ tuned }) => ({ sourceVideoName: tuned.sourceVideo!.video_name }) },
  source_video_end_frame: SOURCE_TRIM,
  source_video_start_frame: SOURCE_TRIM,
  steps: { reports: 'steps', restores: ['steps'], when: ({ policy }) => policy.ui.stepsEditable },
  vae: { reports: 'components', restores: ['vae'] },
  wan_component_source: { reports: 'components', restores: ['componentSourceModel'] },
  wan_guidance_scale_low_noise: {
    absent: { panel: { cfgScaleLowNoise: 2 }, restored: { cfgScaleLowNoise: null } },
    reports: 'cfg',
    restores: ['cfgScaleLowNoise'],
  },
  wan_t5_encoder_model: { reports: 'components', restores: ['wanT5EncoderModel'] },
  wan_transformer_low_noise: { reports: 'components', restores: ['wanLowNoiseModel'] },
  width: SIZE,
};

/** Keyed `<metadata key> <- <source node type>`. */
const VIDEO_RUNTIME_CONTRACT: Readonly<Record<string, ReasonEntry>> = {
  'first_frame_image <- video_frame_extract': {
    ignored:
      "An extension conditions on its source's last frame, extracted when the run executes; recall restores the " +
      'source instead and drops a first frame recorded beside it.',
  },
  'num_frames <- ltx2_duration': {
    consumed:
      'The length the duration head chose, known once the run executes. A real record carries it as an integer, ' +
      'which recall restores like any recorded length and turns Auto duration off (videoRecall.test.ts).',
  },
};

/**
 * Keys only older records carry, still read so those recall; no graph writes them, and each has an old-record test
 * in videoRecall.test.ts. With a value each reads, so a reader that stops reading one shows here as a stale entry.
 */
const LEGACY_READ_ONLY_KEYS: Readonly<Record<string, unknown>> = {
  guidance_scale_low_noise: 3,
  minimax_h3_duration_seconds: 10,
  minimax_h3_transformer_model: { key: 'h3-transformer' },
  transformer_low_noise: { key: 'wan-low-noise' },
  wan_t5_encoder: { key: 'umt5' },
  wan_vae_model: { key: 'wan-vae' },
};

/** Type-exhaustive: a new `VideoRecalledField` fails to compile until it is listed, and so has to be reported. */
const VIDEO_RECALLED_FIELDS = Object.keys({
  cfg: true,
  components: true,
  extendContext: true,
  fps: true,
  frames: true,
  loras: true,
  media: true,
  model: true,
  prompts: true,
  seed: true,
  size: true,
  steps: true,
} satisfies Record<VideoRecalledField, true>) as VideoRecalledField[];

const VIDEO_RECALL_KINDS = Object.keys({
  all: true,
  prompts: true,
  remix: true,
  seed: true,
  size: true,
} satisfies Record<VideoRecallKind, true>) as VideoRecallKind[];

const DEFAULT_KINDS: readonly VideoRecallKind[] = ['all', 'remix'];

// Distinct names per slot so a swapped slot cannot pass; a 10:7 ratio, which no preset has and no canvas keeps once
// snapped, so a media-sized canvas is never also a preset's and only the media say which resolution sized it; and trims
// inside the clips, so a trim is a recorded choice and not the clip's own bounds.
const FIRST_FRAME = { height: 700, image_name: 'first.png', width: 1000 };
const LAST_FRAME = { height: 700, image_name: 'last.png', width: 1000 };
const SOURCE_VIDEO: VideoSourceClip = {
  endFrame: 43,
  fps: 30,
  height: 700,
  numFrames: 120,
  startFrame: 4,
  video_name: 'source.mp4',
  width: 1000,
};
const CONDITIONING_VIDEO = { fps: 30, height: 700, numFrames: 120, video_name: 'conditioning.mp4', width: 1000 };

/** The conditioning each mode needs, as `MODE_INPUTS` in the coverage fixture gives it, with recallable choices made. */
const TUNED_MEDIA: Record<VideoGenerationMode, Partial<VideoSettings>> = {
  'audio-to-video': { conditioningClip: { clip: CONDITIONING_VIDEO, fpsKnown: true, role: 'audio' } },
  extend: { sourceVideo: SOURCE_VIDEO },
  'first-frame': { firstFrameImage: FIRST_FRAME },
  'first-last': { firstFrameImage: FIRST_FRAME, lastFrameImage: LAST_FRAME },
  'last-frame': { lastFrameImage: LAST_FRAME },
  reference: {
    references: [
      {
        clip: {
          endFrame: 50,
          fps: 24,
          height: 480,
          numFrames: 96,
          startFrame: 3,
          video_name: 'reference.mp4',
          width: 832,
        },
        conditioning: 'video',
        kind: 'video',
      },
      { detail: 'match', image: { height: 600, image_name: 'reference.png', width: 800 }, kind: 'image' },
    ],
  },
  txt2vid: {},
  'video-to-audio': { conditioningClip: { clip: CONDITIONING_VIDEO, fpsKnown: true, role: 'video' } },
};

const PANEL_MODEL: MainModelConfig = {
  base: 'wan',
  format: 'diffusers',
  key: 'panel-wan-i2v',
  name: 'Panel Wan I2V',
  type: 'main',
  variant: 'i2v_a14b',
};

/**
 * The panel a recall lands on: another model's, holding a first frame, a LoRA, a prompt and a switched-off negative
 * prompt, so that replacing, clearing and switching on each of them is observable. The seed is fixed for a
 * deterministic run; its mode stays random, so a recalled seed fixing it shows.
 */
const PANEL: VideoWidgetValues = {
  ...createDefaultVideoWidgetValues([PANEL_MODEL]),
  firstFrameImage: { height: 720, image_name: 'panel-frame.png', width: 1280 },
  loras: [{ isEnabled: true, model: PANEL_LORA, weight: 1 }],
  negativePrompt: 'panel negative',
  negativePromptEnabled: false,
  positivePrompt: 'panel prompt',
  seed: 7,
};

/** A panel already on the recorded model, which a size recall fits itself to. */
const panelOn = (model: MainModelConfig): VideoWidgetValues => ({
  ...createDefaultVideoWidgetValues([model]),
  seed: 7,
});

/**
 * Installed alternatives the model transition takes for an empty required slot, listed first so its pick is never the
 * recorded component: otherwise recall would land on that component without reading its key.
 */
const DECOYS: ModelConfig[] = [
  { base: 'minimax-h3', format: 'diffusers', key: 'decoy-h3', name: 'Decoy H3', type: 'main', variant: 'fl2va' },
  {
    base: 'ltx-2',
    format: 'diffusers',
    key: 'decoy-ltx2-components',
    name: 'Decoy LTX-2 components',
    type: 'main',
    variant: 'ltx2_dev',
  },
  { base: 'ltx-2', key: 'decoy-gemma4', name: 'Decoy Gemma-4 encoder', type: 'gemma4_encoder' },
];

/** A concept every family loads, at a weight other than the default; named so no accelerator claims it. */
const tunedLora = (model: MainModelConfig): GenerateLora => ({
  isEnabled: true,
  model: { base: model.base, key: `tuned-lora-${model.base}`, name: 'Contract style LoRA', type: 'lora' },
  weight: 0.75,
});

/**
 * A main slot is offered both shapes: a component source takes a folder, but the low-noise expert and the hybrid
 * base take a single file, which the coverage pool -- it fills only required slots -- never needs.
 */
const componentCandidates = (slot: VideoComponentSlotPolicy): ModelConfig[] =>
  candidatesForSlot(slot).flatMap((candidate) =>
    slot.valueKind === 'main'
      ? [candidate, { ...candidate, format: 'checkpoint', key: `${candidate.key}-single-file` }]
      : [candidate]
  );

/** Every slot the model offers, optional ones included: an empty slot leaves its metadata key unwritten. */
const fillEveryComponentSlot = (model: MainModelConfig, settings: VideoSettings, label: string): VideoSettings => {
  const { slots } = getVideoComponentSectionPolicy(model, settings);
  let filled = settings;

  for (const slot of slots) {
    const context = { model, selectedComponents: filled, settings: filled };
    const candidate =
      filled[slot.key] ?? componentCandidates(slot).find((c) => !slot.filter || slot.filter(c, context));

    filled = { ...filled, [slot.key]: (candidate ?? null) as never };
  }

  expect(
    slots.filter((slot) => !filled[slot.key]).map((slot) => slot.key),
    `${label}: no candidate fills these slots`
  ).toEqual([]);

  return filled;
};

const tunedValuesFor = (testCase: Case, model: MainModelConfig): VideoWidgetValues => {
  const conditioned: VideoSettings = {
    ...getDefaultVideoSettings(model),
    ...TUNED_MEDIA[testCase.mode],
    ...(testCase.autoDuration ? { autoDuration: true, ltx2DurationHeadModel: DURATION_HEAD as never } : {}),
  };
  const { defaults, fps, frames, targetResolutions, ui } = getVideoModelPolicy(model, conditioned);
  const nudged = (visible: boolean, value: number | null, fallback: number | null) =>
    visible && value !== null ? value + 0.5 : fallback;
  const tuned: VideoSettings = {
    ...fillEveryComponentSlot(model, conditioned, testCase.label),
    // The panel runs 16:9.
    aspectRatioId: '9:16',
    audioCfgScale: nudged(ui.audioCfgVisible, defaults.audioCfgScale, conditioned.audioCfgScale),
    cfgScale: ui.cfgVisible ? defaults.cfgScale + 0.5 : conditioned.cfgScale,
    cfgScaleLowNoise: nudged(ui.cfgLowNoiseVisible, defaults.cfgScaleLowNoise, conditioned.cfgScaleLowNoise),
    fps: fps.editable ? fps.defaultValue + 1 : conditioned.fps,
    h3HybridStartBlock: conditioned.h3HybridStartBlock + 5,
    loras: [tunedLora(model)],
    ltx2ExtendContextFrames: ui.extendContext
      ? ui.extendContext.value + ui.extendContext.step
      : conditioned.ltx2ExtendContextFrames,
    modalityScale: nudged(ui.modalityVisible, defaults.modalityScale, conditioned.modalityScale),
    // Non-empty: an empty recorded negative prompt deliberately leaves the panel's own.
    negativePrompt: 'tuned negative',
    negativePromptEnabled: true,
    numFrames:
      frames.kind === 'grid'
        ? frames.defaultValue + frames.step
        : frames.choices[frames.choices.indexOf(frames.defaultValue) + 1]!,
    positivePrompt: 'tuned prompt',
    seed: 4242,
    seedMode: 'fixed',
    stgScale: nudged(ui.stgVisible, defaults.stgScale, conditioned.stgScale),
    steps: ui.stepsEditable ? defaults.steps + 1 : conditioned.steps,
    // The last single-stage resolution besides the default: on Wan the extrapolated 1080p, whose short side snaps off
    // the grid, so a media-sized canvas there cannot be resolved from its own shape.
    targetResolution:
      testCase.targetResolution ??
      targetResolutions
        .filter((option) => option.id !== defaults.targetResolution && option.stages === undefined)
        .at(-1)!.id,
  };

  expect(resolveVideoMode(tuned), `${testCase.label}: the tuned media resolve to another mode`).toBe(testCase.mode);
  expect(getVideoValidationReasons(model, tuned), `${testCase.label}: the tuned run is not satisfiable`).toEqual([]);

  return { ...tuned, model };
};

interface RecordedCase extends RecordedRun {
  catalog: readonly ModelConfig[];
  label: string;
  runtimeSources: Record<string, string>;
}

const recordedCases = new Map<string, RecordedCase>();

const recorded = (testCase: Case): RecordedCase => {
  const cached = recordedCases.get(testCase.label);

  if (cached) {
    return cached;
  }

  const model = createModel(testCase);
  const tuned = tunedValuesFor(testCase, model);
  const record = recordOf(compileVideoGraph(tuned, model), tuned);
  const runtimeSources = Object.fromEntries(
    Object.entries(record).flatMap(([key, value]) => {
      const source = runtimeSourceOf(value);

      return source === null ? [] : [[key, source]];
    })
  );

  if (getVideoDimensions(model, tuned)?.source !== 'aspect-ratio') {
    // A media-sized canvas that is also a preset's recalls as that preset, which would hide whether recall recovers
    // the target resolution from the media's shape.
    expect(
      getVideoSizeRecalls(model, record.width as number, record.height as number),
      `${testCase.label}: the media-sized canvas is also a preset's`
    ).toEqual([]);
  }

  const components = getVideoComponentSectionPolicy(model, tuned).slots.flatMap((slot) =>
    tuned[slot.key] ? [tuned[slot.key] as ModelConfig] : []
  );
  const recordedCase: RecordedCase = {
    catalog: [...DECOYS, PANEL_MODEL, model, ...components, ...tuned.loras.map((lora) => lora.model)],
    label: testCase.label,
    model,
    policy: getVideoModelPolicy(model, tuned),
    runtimeSources,
    tuned,
    written: record,
  };

  recordedCases.set(testCase.label, recordedCase);

  return recordedCase;
};

const build = (
  run: RecordedCase,
  kind: VideoRecallKind,
  metadata: Record<string, unknown> = run.written,
  currentValues: VideoWidgetValues = PANEL
) => buildVideoRecallSettings({ currentValues, kind, metadata, models: run.catalog });

/**
 * A recall as the executor applies it: a full one hydrates the media it names -- here the run's own, as the gallery
 * returns them -- and then settles a media-sized canvas's target resolution against them.
 */
const recall = (...args: Parameters<typeof build>) => {
  const [run, kind] = args;
  const result = build(...args);

  if (!result || (kind !== 'all' && kind !== 'remix')) {
    return result;
  }

  const { mediaNames, values } = result;
  const hydrated: VideoWidgetValues = {
    ...values,
    conditioningClip: mediaNames.conditioningClip ? run.tuned.conditioningClip : values.conditioningClip,
    firstFrameImage: mediaNames.firstFrameName ? run.tuned.firstFrameImage : values.firstFrameImage,
    lastFrameImage: mediaNames.lastFrameName ? run.tuned.lastFrameImage : values.lastFrameImage,
    references: mediaNames.references.length > 0 ? run.tuned.references : values.references,
    sourceVideo: mediaNames.sourceVideoName ? run.tuned.sourceVideo : values.sourceVideo,
  };

  return { ...result, ...settleMediaSizedRecall({ ...result, values: hydrated }) };
};

const without = (record: Record<string, unknown>, key: string): Record<string, unknown> => {
  const { [key]: _removed, ...rest } = record;

  return rest;
};

const pick = <T extends object, K extends keyof T>(source: T, keys: readonly K[]): Pick<T, K> =>
  Object.fromEntries(keys.map((key) => [key, source[key]])) as Pick<T, K>;

const lookup = <T>(table: Readonly<Record<string, T>>, key: string): T | undefined =>
  Object.hasOwn(table, key) ? table[key] : undefined;

/** The contract key a recorded value is classified under: a value wired in at run time is classified with its source. */
const contractKeyOf = (run: RecordedCase, key: string): string =>
  run.runtimeSources[key] === undefined ? key : `${key} <- ${run.runtimeSources[key]}`;

const entryOf = (run: RecordedCase, key: string): Entry | undefined =>
  run.runtimeSources[key] === undefined
    ? lookup(VIDEO_RECALL_CONTRACT, key)
    : lookup(VIDEO_RUNTIME_CONTRACT, contractKeyOf(run, key));

/** Each recorded key with the entry governing it; G1 reports the keys without one. */
const entriesOf = (run: RecordedCase): [string, Entry][] =>
  Object.keys(run.written).flatMap((key): [string, Entry][] => {
    const entry = entryOf(run, key);

    return entry ? [[key, entry]] : [];
  });

const isRestoring = (entry: Entry, run: RecordedRun): entry is RestoresEntry =>
  'restores' in entry && (entry.when?.(run) ?? true);

const restoredOn = (entry: RestoresEntry, run: RecordedRun): WidgetValueKeys =>
  typeof entry.restores === 'function' ? entry.restores(run) : entry.restores;

const expectedValues = (entry: RestoresEntry, run: RecordedRun): Partial<VideoWidgetValues> => ({
  ...pick(run.tuned, restoredOn(entry, run)),
  ...entry.expected?.(run),
});

/** The values the verb applies from this record: those of every entry restored on this run that the verb carries. */
const patchFor = (run: RecordedCase, kind: VideoRecallKind): Partial<VideoWidgetValues> =>
  Object.assign(
    {},
    ...entriesOf(run).flatMap(([, entry]) =>
      isRestoring(entry, run) && (entry.kinds ?? DEFAULT_KINDS).includes(kind) ? [expectedValues(entry, run)] : []
    )
  );

const NO_MEDIA_NAMES: VideoRecallMediaNames = {
  conditioningClip: null,
  firstFrameName: null,
  lastFrameName: null,
  references: [],
  sourceVideoName: null,
  sourceVideoTrim: null,
};

const ALL_CASES = [...cases, ...AUTO_DURATION_CASES];

describe('video recall contract', () => {
  it.each(ALL_CASES)('G1 gives every key $label records a contract entry', (testCase) => {
    const run = recorded(testCase);
    const unaccounted = Object.keys(run.written).flatMap((key) =>
      entryOf(run, key)
        ? []
        : [
            `\`${contractKeyOf(run, key)}\` has no entry in ${
              run.runtimeSources[key] === undefined ? 'VIDEO_RECALL_CONTRACT' : 'VIDEO_RUNTIME_CONTRACT'
            }`,
          ]
    );

    expect(unaccounted, `G1 ${testCase.label}`).toEqual([]);
  });

  it.each(ALL_CASES)('G2 restores every recorded value of $label, each through its own key', (testCase) => {
    const run = recorded(testCase);
    const result = recall(run, 'all');
    const expectedMedia = { ...NO_MEDIA_NAMES };

    expect(result, `G2 ${testCase.label}: nothing recalled`).not.toBeNull();

    for (const [key, entry] of entriesOf(run)) {
      if ('media' in entry) {
        const ablated = recall(run, 'all', without(run.written, key));
        const media = entry.media(run);

        Object.assign(expectedMedia, media);
        for (const field of Object.keys(media) as (keyof VideoRecallMediaNames)[]) {
          expect(result?.mediaNames[field], `G2 ${testCase.label}: \`${key}\` -> mediaNames.${field}`).toEqual(
            media[field]
          );
          expect(
            ablated?.mediaNames[field],
            `G2 ${testCase.label}: \`${key}\` is vacuous for mediaNames.${field}; recall without it names the same`
          ).not.toEqual(media[field]);
        }
        expect(result?.fields, `G2 ${testCase.label}: \`${key}\` is not reported`).toContain('media');
        continue;
      }
      if (!isRestoring(entry, run)) {
        continue;
      }

      const ablated = recall(run, 'all', without(run.written, key));
      const expected = expectedValues(entry, run);

      for (const field of restoredOn(entry, run)) {
        expect(result?.values[field], `G2 ${testCase.label}: \`${key}\` -> ${field}`).toEqual(expected[field]);
        expect(
          ablated?.values[field],
          `G2 ${testCase.label}: \`${key}\` is vacuous for ${field}; recall without it already lands there`
        ).not.toEqual(expected[field]);
      }
      expect(result?.fields, `G2 ${testCase.label}: \`${key}\` is not reported as ${entry.reports}`).toContain(
        entry.reports
      );
    }

    expect(result?.mediaNames, `G2 ${testCase.label}: the media recall names`).toEqual(expectedMedia);
    // A full recall clears the panel's media wholesale; the executor hydrates `mediaNames` into these slots.
    const built = build(run, 'all');

    expect(
      built &&
        pick(built.values, ['conditioningClip', 'firstFrameImage', 'lastFrameImage', 'references', 'sourceVideo']),
      `G2 ${testCase.label}: the panel's media survive the recall`
    ).toEqual({
      conditioningClip: null,
      firstFrameImage: null,
      lastFrameImage: null,
      references: [],
      sourceVideo: null,
    });
  });

  it.each(ALL_CASES)('G3 applies nothing from the keys of $label it leaves alone', (testCase) => {
    const run = recorded(testCase);
    const applied = (result: ReturnType<typeof recall>) => ({ mediaNames: result?.mediaNames, values: result?.values });
    const full = applied(recall(run, 'all'));

    for (const [key, entry] of entriesOf(run)) {
      if ('ignored' in entry || ('restores' in entry && !isRestoring(entry, run))) {
        expect(
          applied(recall(run, 'all', without(run.written, key))),
          `G3 ${testCase.label}: \`${key}\` is held inert but changes what recall applies`
        ).toEqual(full);
      }
    }
  });

  it.each(ALL_CASES)('G4 offers every recall verb on $label, each applying exactly its share', (testCase) => {
    const run = recorded(testCase);
    const all = recall(run, 'all')!;
    const remix = recall(run, 'remix');
    const allOnly = entriesOf(run).flatMap(([, entry]) =>
      isRestoring(entry, run) && !(entry.kinds ?? DEFAULT_KINDS).includes('remix') ? [entry] : []
    );

    expect(getVideoRecallCapabilities(run.written), `G4 ${testCase.label}: capabilities`).toEqual({
      all: true,
      prompts: true,
      remix: true,
      seed: true,
      size: true,
    });
    expect(remix?.values, `G4 ${testCase.label}: remix`).toEqual({
      ...all.values,
      ...pick(
        PANEL,
        allOnly.flatMap((entry) => restoredOn(entry, run))
      ),
    });
    expect(remix?.mediaNames, `G4 ${testCase.label}: remix media`).toEqual(all.mediaNames);
    expect(remix?.fields, `G4 ${testCase.label}: remix fields`).toEqual(
      all.fields.filter((field) => !allOnly.some((entry) => entry.reports === field))
    );

    for (const kind of ['prompts', 'seed'] as const) {
      expect(recall(run, kind)?.values, `G4 ${testCase.label}: ${kind}`).toEqual({ ...PANEL, ...patchFor(run, kind) });
    }

    // Fitted to the panel's own model and media, so a panel already on the recorded model and holding its media.
    const panel = { ...panelOn(run.model), ...TUNED_MEDIA[testCase.mode] };
    const size = patchFor(run, 'size');

    expect(recall(run, 'size', run.written, panel)?.values ?? null, `G4 ${testCase.label}: size`).toEqual(
      Object.keys(size).length > 0 ? { ...panel, ...size } : null
    );
  });

  it.each(ALL_CASES)('G8 restores what the absence of a key means on a full recall of $label', (testCase) => {
    const run = recorded(testCase);

    for (const [key, entry] of entriesOf(run)) {
      if (!('restores' in entry) || !entry.absent) {
        continue;
      }

      const { panel, restored } = entry.absent;
      const result = recall(run, 'all', without(run.written, key), { ...panelOn(run.model), ...panel });

      expect(
        result && pick(result.values, Object.keys(restored) as (keyof VideoWidgetValues)[]),
        `G8 ${testCase.label}: \`${key}\` absent`
      ).toEqual(restored);
    }
  });

  it('G7 has no entry no recorded case exercises', () => {
    const exercised = new Set(
      ALL_CASES.flatMap((testCase) => {
        const run = recorded(testCase);

        return Object.keys(run.written).map((key) => contractKeyOf(run, key));
      })
    );

    expect(
      [...Object.keys(VIDEO_RECALL_CONTRACT), ...Object.keys(VIDEO_RUNTIME_CONTRACT)].filter(
        (key) => !exercised.has(key)
      ),
      'G7: entries no case records'
    ).toEqual([]);
  });

  it('keeps the legacy read-only keys out of every record, and still reads each', () => {
    const recordedKeys = new Set(ALL_CASES.flatMap((testCase) => Object.keys(recorded(testCase).written)));
    const base = { generation_mode: 'minimax_h3_t2v' };

    for (const [key, value] of Object.entries(LEGACY_READ_ONLY_KEYS)) {
      expect(recordedKeys.has(key), `\`${key}\` is written again; give it a contract entry instead`).toBe(false);
      expect(parseVideoRecallRecord({ ...base, [key]: value }), `\`${key}\` is no longer read`).not.toEqual(
        parseVideoRecallRecord(base)
      );
    }
  });

  describe('recall notice copy', () => {
    const englishCatalog = Object.values(
      import.meta.glob('../../../public/locales/en.json', { eager: true, import: 'default' })
    )[0] as Record<string, unknown>;
    const english = i18n.createInstance();
    const t = ((key: string, values?: Record<string, unknown>) => english.t(key, values)) as TFunction;

    beforeAll(async () => {
      await english.init({
        initAsync: false,
        interpolation: { escapeValue: false },
        lng: 'en',
        resources: { en: { translation: englishCatalog } },
      });
    });

    // The labels are looked up by a key built at run time, which the static translation-key scan cannot see.
    it('G5 reports every recalled field on some recorded case, under an English label', () => {
      const reported = new Set(ALL_CASES.flatMap((testCase) => recall(recorded(testCase), 'all')?.fields ?? []));

      expect(
        VIDEO_RECALLED_FIELDS.filter((field) => !reported.has(field)),
        'G5: fields no case reports'
      ).toEqual([]);
      for (const field of VIDEO_RECALLED_FIELDS) {
        expect(getVideoRecallMessage([field], t), `G5: \`${field}\` has no label`).not.toContain('widgets.gallery');
      }
    });

    it('titles every recall verb in English', () => {
      for (const kind of VIDEO_RECALL_KINDS) {
        expect(getVideoRecallTitle(kind, t), `\`${kind}\` has no title`).not.toContain('widgets.gallery');
      }
    });
  });
});
