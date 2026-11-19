/**
 * The registry-driven video case matrix: every supported base, variant, generation mode, model shape and multi-stage
 * preset, and the compile each case runs. Shared by `videoGraphCoverage.test.ts`, whose committed fixture records
 * what these cases emit, and the recall contract in `workbench/image-actions/videoRecallContract.test.ts`, which
 * recalls what they record.
 */

import type { GenerationModelCatalogItem as ModelConfig, MainModelConfig } from '@features/generation/contracts';
import type { BackendGraphContract } from '@features/generation/core/contracts';

import { expect } from 'vitest';

import type { VideoGenerationMode, VideoSettings, VideoSourceClip, VideoTargetResolution } from './types';
import type { VideoComponentPolicyContext, VideoComponentSlotPolicy, VideoComponentValueKey } from './videoPolicies';

import { compileVideoGraph } from './graph';
import {
  getDefaultVideoSettings,
  getVideoComponentSectionPolicy,
  getVideoValidationReasons,
  isAutoDurationActive,
  isTwoStageSupportedForMode,
  isVideoModelSelectable,
  SUPPORTED_VIDEO_BASES,
  VIDEO_GENERATION,
} from './videoPolicies';

const IMAGE = { height: 704, image_name: 'frame.png', width: 1248 };
const CLIP: VideoSourceClip = {
  endFrame: 47,
  fps: 24,
  height: 704,
  numFrames: 48,
  startFrame: 0,
  video_name: 'clip.mp4',
  width: 1248,
};

const { endFrame: _end, startFrame: _start, ...CLIP_REF } = CLIP;

/** The conditioning each mode needs; `resolveVideoMode` infers the mode back from it. */
const MODE_INPUTS: Record<VideoGenerationMode, Partial<VideoSettings>> = {
  'audio-to-video': { conditioningClip: { clip: CLIP_REF, fpsKnown: true, role: 'audio' } },
  extend: { sourceVideo: CLIP },
  'first-frame': { firstFrameImage: IMAGE },
  'first-last': { firstFrameImage: IMAGE, lastFrameImage: IMAGE },
  'last-frame': { lastFrameImage: IMAGE },
  reference: { references: [{ detail: 'max', image: IMAGE, kind: 'image' }] },
  txt2vid: {},
  'video-to-audio': { conditioningClip: { clip: CLIP_REF, fpsKnown: true, role: 'video' } },
};

/**
 * Candidate components for the slot filters to choose from — a search over a pool rather than a
 * hand-written model per slot, so the test agrees with the policy rather than with itself.
 */
const CANDIDATE_BASES = ['any', ...SUPPORTED_VIDEO_BASES] as const;
const CANDIDATE_VARIANTS = [undefined, 'fl2va', 'ref2va', 't2v_a14b', 'i2v_a14b', 'ti2v_5b', 'ltx2_dev'] as const;
const CANDIDATE_LATENT_CHANNELS = [undefined, 16, 48] as const;

export const candidatesForSlot = (slot: VideoComponentSlotPolicy): ModelConfig[] => {
  const candidates: ModelConfig[] = [];

  for (const type of slot.modelTypes) {
    for (const base of CANDIDATE_BASES) {
      for (const variant of CANDIDATE_VARIANTS) {
        for (const latentChannels of type === 'vae' ? CANDIDATE_LATENT_CHANNELS : [undefined]) {
          candidates.push({
            base,
            // A component source has to be a bundled install; every other slot takes a
            // single file, and the H3/LTX-2 encoder slots do not care either way.
            format: slot.valueKind === 'main' ? 'diffusers' : 'checkpoint',
            key: `${base}-${type}-${variant ?? 'novariant'}${latentChannels ? `-${latentChannels}` : ''}`,
            name: `${base} ${type} ${variant ?? ''}`.trim(),
            type,
            variant: variant ?? null,
            ...(latentChannels ? { latent_channels: latentChannels } : {}),
          } as ModelConfig);
        }
      }
    }
  }

  return candidates;
};

const buildContext = (
  model: MainModelConfig,
  settings: VideoSettings,
  slots: readonly VideoComponentSlotPolicy[]
): VideoComponentPolicyContext => {
  const selectedComponents = {} as VideoComponentPolicyContext['selectedComponents'];

  for (const slot of slots) {
    selectedComponents[slot.key] = settings[slot.key] as never;
  }

  return { model, selectedComponents, settings };
};

/**
 * Fill every required slot with something its own filter accepts. Iterated to a fixpoint because
 * slots are interdependent: picking a component source can make another slot stop being required.
 */
const satisfyRequiredComponents = (
  model: MainModelConfig,
  initial: VideoSettings
): { settings: VideoSettings; filled: VideoComponentValueKey[] } => {
  let settings = initial;
  const filled: VideoComponentValueKey[] = [];

  for (let pass = 0; pass < 5; pass++) {
    const { slots } = getVideoComponentSectionPolicy(model, settings);
    const context = buildContext(model, settings, slots);
    let changed = false;

    for (const slot of slots) {
      if (!slot.required?.(context) || settings[slot.key]) {
        continue;
      }

      const candidate = candidatesForSlot(slot).find((c) => !slot.filter || slot.filter(c, context));

      if (candidate) {
        settings = { ...settings, [slot.key]: candidate as never };
        filled.push(slot.key);
        changed = true;
      }
    }

    if (!changed) {
      break;
    }
  }

  return { filled: filled.sort(), settings };
};

export interface Case {
  base: string;
  variant: string;
  mode: VideoGenerationMode;
  format: 'diffusers' | 'checkpoint';
  /** A preset to override the variant's default with; absent runs the default. */
  targetResolution?: VideoTargetResolution;
  /** Fill the optional LTX-2 duration head slot and let it choose the length. */
  autoDuration?: true;
  label: string;
}

export const createModel = (testCase: Case): MainModelConfig =>
  ({
    base: testCase.base,
    format: testCase.format,
    key: `${testCase.base}-${testCase.variant}-${testCase.format}`,
    name: `${testCase.base} ${testCase.variant} (${testCase.format})`,
    type: 'main',
    variant: testCase.variant,
  }) as MainModelConfig;

/**
 * Both model shapes per variant: a bundled install supplies its own components, a single file
 * forces the component-source path, and the two produce different graphs. A shape the panel would
 * not let the user pick at all is dropped — which shapes those are is the policy's answer
 * (`isVideoModelSelectable`), not a list kept here: H3's Ref2VA folder install is supported state
 * but not runnable as the model, and hard-coding that would go stale.
 */
export const cases: Case[] = SUPPORTED_VIDEO_BASES.flatMap((base) =>
  Object.entries(VIDEO_GENERATION[base].variants).flatMap(([variant, config]) => {
    // A multi-stage preset builds a different graph -- more denoise passes, and the nodes between
    // them -- so each one is its own case. Every one of them, not just the first: the fixture
    // records the literal values written into each field as well as the node types, and the widest
    // canvas is exactly the one a future bound on those fields would reject.
    const presets: (VideoTargetResolution | undefined)[] = [
      undefined,
      ...config.targetResolutions.filter((option) => option.stages !== undefined).map((option) => option.id),
    ];

    return config.modes.flatMap((mode) =>
      presets
        .filter((targetResolution) => {
          // A mode the policy refuses to run two-stage is not an uncovered case, it is a
          // combination that does not exist -- asked of the policy so the two cannot drift.
          const option = config.targetResolutions.find((entry) => entry.id === targetResolution);

          return option?.stages !== 2 || isTwoStageSupportedForMode(mode);
        })
        .flatMap((targetResolution) =>
          (['diffusers', 'checkpoint'] as const)
            .map((format) => ({
              base,
              format,
              label: `${base} / ${variant} / ${mode} / ${format}${targetResolution ? ` / ${targetResolution}` : ''}`,
              mode,
              ...(targetResolution ? { targetResolution } : {}),
              variant,
            }))
            .filter((testCase) => isVideoModelSelectable(createModel(testCase)))
        )
    );
  })
);

/**
 * Auto duration is an optional slot plus a flag, so the matrix above -- which fills only *required*
 * slots -- never reaches the `ltx2_duration` node. Added here rather than as another dimension of
 * that matrix because it only affects one base. Every preset is kept, including the two-stage ones:
 * those build a second denoise with a `num_frames` of its own, so restricting this to the default
 * preset is exactly how a graph that dies after the base pass stays invisible.
 */
export const DURATION_HEAD = {
  base: 'ltx-2',
  key: 'ltx2-duration-head',
  name: 'LTX-2 duration head',
  type: 'ltx2_duration_head',
} as ModelConfig;

export const AUTO_DURATION_CASES: Case[] = cases
  .filter((testCase) => testCase.base === 'ltx-2')
  .filter((testCase) =>
    // Asked of the policy, not of a list of modes kept here, so the two cannot drift.
    isAutoDurationActive({
      autoDuration: true,
      conditioningClip: null,
      firstFrameImage: null,
      lastFrameImage: null,
      ltx2DurationHeadModel: DURATION_HEAD,
      references: [],
      sourceVideo: null,
      ...MODE_INPUTS[testCase.mode],
    } as unknown as VideoSettings)
  )
  .map((testCase) => ({ ...testCase, autoDuration: true, label: `${testCase.label} / auto duration` }));

export const compileForCase = (testCase: Case): { filled: VideoComponentValueKey[]; graph: BackendGraphContract } => {
  const model = createModel(testCase);
  const { filled, settings } = satisfyRequiredComponents(model, {
    ...getDefaultVideoSettings(model),
    ...MODE_INPUTS[testCase.mode],
    positivePrompt: 'a test prompt',
    seed: 1,
    seedMode: 'fixed',
    ...(testCase.targetResolution ? { targetResolution: testCase.targetResolution } : {}),
    ...(testCase.autoDuration ? { autoDuration: true, ltx2DurationHeadModel: DURATION_HEAD as never } : {}),
  });

  // Compiling an invalid selection throws only the first reason; asserting here reports every
  // unmet requirement at once, and doubles as the check that the policy is satisfiable at all.
  expect(getVideoValidationReasons(model, settings), `${testCase.label} is not satisfiable`).toEqual([]);

  return { filled, graph: compileVideoGraph(settings, model).backendGraph };
};
