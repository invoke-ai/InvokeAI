import type { DetailerQuality } from 'features/controlLayers/store/types';

export type DetailerTargetProfile = 'localized' | 'person';

export type DetailerDenoiseConfig = {
  cfgScale: number;
  steps: number;
  strength: number;
  edgeRadius: number;
};

export type GroundedSamDetailerRuntimeConfig = DetailerDenoiseConfig & {
  targetProfile: DetailerTargetProfile;
  targetSize: number;
  maxUpscale: number;
  maxProcessSize: number;
  denoiseMaskExpand: number;
  denoiseMaskFeather: number;
  pasteMaskExpand: number;
  pasteMaskFeather: number;
  denoiseMaskContract: number;
  pasteMaskContract: number;
  preventDownscale: boolean;
};

export type GroundedSamDetectorPromptConfig = {
  detectorPrompt: string;
  labelPriority: string;
};

type DetailerRuntimeConfigParams = {
  detailerQuality: DetailerQuality;
  detailerTargetSize: number;
  detailerMaxUpscale: number;
  detailerMaxProcessSize: number;
  detailerDenoiseMaskExpand: number;
  detailerDenoiseMaskFeather: number;
  detailerPasteMaskExpand: number;
  detailerPasteMaskFeather: number;
  detailerCfgScale: number;
  detailerSteps: number;
  detailerStrength: number;
};

const PERSON_TARGET_PROMPTS = new Set(['person', 'body', 'full body', 'human', 'character', 'figure']);

const PERSON_RUNTIME_PRESETS: Record<
  DetailerQuality,
  {
    processSize: number;
    maxUpscale: number;
    strength: number;
    steps: number;
    cfgScaleCap: number;
  }
> = {
  fast: {
    processSize: 1024,
    maxUpscale: 4,
    strength: 0.1,
    steps: 8,
    cfgScaleCap: 4.5,
  },
  balanced: {
    processSize: 1024,
    maxUpscale: 8,
    strength: 0.14,
    steps: 14,
    cfgScaleCap: 4.5,
  },
  high: {
    processSize: 1536,
    maxUpscale: 12,
    strength: 0.18,
    steps: 18,
    cfgScaleCap: 5,
  },
};

const getNormalizedTargetPrompt = (targetPrompt: string) => targetPrompt.trim().toLocaleLowerCase().replace(/\s+/g, ' ');

export const getDetailerTargetProfile = (targetPrompt: string): DetailerTargetProfile =>
  PERSON_TARGET_PROMPTS.has(getNormalizedTargetPrompt(targetPrompt)) ? 'person' : 'localized';

export const getGroundedSamDetectorPromptConfig = (targetPrompt: string): GroundedSamDetectorPromptConfig => {
  const trimmedTargetPrompt = targetPrompt.trim() || 'face';
  const normalizedTargetPrompt = getNormalizedTargetPrompt(trimmedTargetPrompt);

  if (normalizedTargetPrompt === 'face') {
    return { detectorPrompt: 'face', labelPriority: 'face' };
  }

  if (normalizedTargetPrompt === 'head') {
    return { detectorPrompt: 'head', labelPriority: 'head' };
  }

  if (normalizedTargetPrompt === 'hand' || normalizedTargetPrompt === 'hands') {
    return { detectorPrompt: 'hands', labelPriority: 'hands' };
  }

  if (normalizedTargetPrompt === 'person' || normalizedTargetPrompt === 'body') {
    return { detectorPrompt: 'person', labelPriority: 'person' };
  }

  return { detectorPrompt: trimmedTargetPrompt, labelPriority: trimmedTargetPrompt };
};

export const getGroundedSamDetailerRuntimeConfig = (
  params: DetailerRuntimeConfigParams,
  targetPrompt: string
): GroundedSamDetailerRuntimeConfig => {
  const targetProfile = getDetailerTargetProfile(targetPrompt);

  if (targetProfile === 'localized') {
    return {
      targetProfile,
      targetSize: params.detailerTargetSize,
      maxUpscale: params.detailerMaxUpscale,
      maxProcessSize: params.detailerMaxProcessSize,
      denoiseMaskExpand: params.detailerDenoiseMaskExpand,
      denoiseMaskFeather: params.detailerDenoiseMaskFeather,
      pasteMaskExpand: params.detailerPasteMaskExpand,
      pasteMaskFeather: params.detailerPasteMaskFeather,
      denoiseMaskContract: 0,
      pasteMaskContract: 0,
      preventDownscale: false,
      cfgScale: params.detailerCfgScale,
      steps: params.detailerSteps,
      strength: params.detailerStrength,
      edgeRadius: params.detailerDenoiseMaskFeather,
    };
  }

  const preset = PERSON_RUNTIME_PRESETS[params.detailerQuality];

  return {
    targetProfile,
    targetSize: preset.processSize,
    maxUpscale: preset.maxUpscale,
    maxProcessSize: preset.processSize,
    denoiseMaskExpand: 0,
    denoiseMaskFeather: 2,
    pasteMaskExpand: 0,
    pasteMaskFeather: 4,
    denoiseMaskContract: 4,
    pasteMaskContract: 2,
    preventDownscale: true,
    cfgScale: Math.min(params.detailerCfgScale, preset.cfgScaleCap),
    steps: preset.steps,
    strength: preset.strength,
    edgeRadius: 2,
  };
};
