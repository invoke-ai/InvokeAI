/** SDXL Refiner policy: which models may fill the slot, its value ranges, and the denoise hand-over. */

import type { GenerateModelConfig, GenerateSettings, MainModelConfig } from './types';

export const DEFAULT_REFINER_STEPS = 20;
export const DEFAULT_REFINER_CFG_SCALE = 7.5;
export const DEFAULT_REFINER_SCHEDULER = 'euler';
export const DEFAULT_REFINER_POSITIVE_AESTHETIC_SCORE = 6;
export const DEFAULT_REFINER_NEGATIVE_AESTHETIC_SCORE = 2.5;
export const DEFAULT_REFINER_START = 0.8;

export const MIN_REFINER_STEPS = 1;
export const MIN_REFINER_CFG_SCALE = 1;
export const MIN_REFINER_AESTHETIC_SCORE = 1;
export const MAX_REFINER_AESTHETIC_SCORE = 10;
export const MIN_REFINER_START = 0;
export const MAX_REFINER_START = 1;

export type RefinerSettings = Pick<
  GenerateSettings,
  | 'refinerModel'
  | 'refinerSteps'
  | 'refinerCfgScale'
  | 'refinerScheduler'
  | 'refinerPositiveAestheticScore'
  | 'refinerNegativeAestheticScore'
  | 'refinerStart'
>;

/** The refiner is its own model family; only it may fill the slot, and never the main-model picker. */
export const isRefinerModel = (model: { base: string; type: string } | null | undefined): boolean =>
  model?.type === 'main' && model.base === 'sdxl-refiner';

/** Only the SDXL base hands its latents to a refiner. */
export const isRefinerSupported = (model: GenerateModelConfig | undefined): boolean =>
  model?.type === 'main' && model.base === 'sdxl';

/**
 * The refiner that will run: a refiner model selected on an SDXL main model. A start of 1 leaves the refiner no
 * schedule, so it is treated as off rather than loading a model for an empty pass.
 */
export const getActiveRefiner = (
  settings: Pick<GenerateSettings, 'refinerModel' | 'refinerStart'>,
  model: GenerateModelConfig | undefined
): MainModelConfig | null =>
  isRefinerSupported(model) && isRefinerModel(settings.refinerModel) && settings.refinerStart < MAX_REFINER_START
    ? settings.refinerModel
    : null;

/**
 * The base model's denoise range when a refiner finishes the schedule. An image-to-image start past the hand-over
 * would leave the base model nothing to do, so the start is capped at the hand-over point (legacy parity).
 */
export const getRefinerDenoisingSplit = (
  refinerStart: number,
  denoisingStart = 0
): { denoising_start: number; denoising_end: number } => ({
  denoising_start: Math.min(refinerStart, denoisingStart),
  denoising_end: refinerStart,
});

/**
 * The single owner of the base denoise range. `denoisingStart` is the strength-derived start (0 for text-to-image);
 * with an active refiner the base pass ends at the hand-over, otherwise it runs to the end of the schedule.
 */
export const getBaseDenoisingRange = (
  settings: Pick<GenerateSettings, 'refinerModel' | 'refinerStart'>,
  model: GenerateModelConfig,
  denoisingStart: number
): { denoising_start: number; denoising_end: number } =>
  getActiveRefiner(settings, model)
    ? getRefinerDenoisingSplit(settings.refinerStart, denoisingStart)
    : { denoising_end: 1, denoising_start: denoisingStart };

export const getRefinerValidationReasons = (
  model: GenerateModelConfig,
  settings: RefinerSettings & Pick<GenerateSettings, 'pidMode'>
): string[] => {
  if (!settings.refinerModel || !isRefinerSupported(model) || settings.refinerStart === MAX_REFINER_START) {
    return [];
  }

  if (!isRefinerModel(settings.refinerModel)) {
    return [`${settings.refinerModel.name} is not an SDXL Refiner model. Pick a refiner or clear the slot.`];
  }

  const reasons: string[] = [];

  // PiD replaces the VAE decode the refiner's second denoise stage feeds; the two cannot share a graph.
  if (settings.pidMode !== 'off' && settings.refinerStart < MAX_REFINER_START) {
    reasons.push('PiD decode is not compatible with the SDXL Refiner. Turn PiD off or clear the refiner model.');
  }

  if (!Number.isInteger(settings.refinerSteps) || settings.refinerSteps < MIN_REFINER_STEPS) {
    reasons.push(`Refiner steps must be a whole number of at least ${MIN_REFINER_STEPS}.`);
  }

  if (!Number.isFinite(settings.refinerCfgScale) || settings.refinerCfgScale < MIN_REFINER_CFG_SCALE) {
    reasons.push(`Refiner CFG must be at least ${MIN_REFINER_CFG_SCALE}.`);
  }

  if (
    !Number.isFinite(settings.refinerStart) ||
    settings.refinerStart < MIN_REFINER_START ||
    settings.refinerStart > MAX_REFINER_START
  ) {
    reasons.push(`Refiner start must be between ${MIN_REFINER_START} and ${MAX_REFINER_START}.`);
  }

  for (const [label, value] of [
    ['positive', settings.refinerPositiveAestheticScore],
    ['negative', settings.refinerNegativeAestheticScore],
  ] as const) {
    if (!Number.isFinite(value) || value < MIN_REFINER_AESTHETIC_SCORE || value > MAX_REFINER_AESTHETIC_SCORE) {
      reasons.push(
        `Refiner ${label} aesthetic score must be between ${MIN_REFINER_AESTHETIC_SCORE} and ${MAX_REFINER_AESTHETIC_SCORE}.`
      );
    }
  }

  return reasons;
};
