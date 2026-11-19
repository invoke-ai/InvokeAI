/** The video case matrix, shared by `videoGraphCoverage.test.ts` and `videoRecallContract.test.ts`. */

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

/** A pool the slot filters choose from, so the test agrees with the policy rather than with itself. */
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

/** Iterated to a fixpoint: picking a component source can make another slot stop being required. */
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
  targetResolution?: VideoTargetResolution;
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

/** Both model shapes per variant, minus those `isVideoModelSelectable` refuses. */
export const cases: Case[] = SUPPORTED_VIDEO_BASES.flatMap((base) =>
  Object.entries(VIDEO_GENERATION[base].variants).flatMap(([variant, config]) => {
    // Every multi-stage preset is its own case: the widest canvas is the one a future field bound would reject.
    const presets: (VideoTargetResolution | undefined)[] = [
      undefined,
      ...config.targetResolutions.filter((option) => option.stages !== undefined).map((option) => option.id),
    ];

    return config.modes.flatMap((mode) =>
      presets
        .filter((targetResolution) => {
          // Asked of the policy so the two cannot drift.
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

/** Auto duration is an optional slot the required-only matrix never fills; every preset is kept. */
export const DURATION_HEAD = {
  base: 'ltx-2',
  key: 'ltx2-duration-head',
  name: 'LTX-2 duration head',
  type: 'ltx2_duration_head',
} as ModelConfig;

export const AUTO_DURATION_CASES: Case[] = cases
  .filter((testCase) => testCase.base === 'ltx-2')
  .filter((testCase) =>
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

  // Reports every unmet requirement at once, where compiling throws only the first.
  expect(getVideoValidationReasons(model, settings), `${testCase.label} is not satisfiable`).toEqual([]);

  return { filled, graph: compileVideoGraph(settings, model).backendGraph };
};
