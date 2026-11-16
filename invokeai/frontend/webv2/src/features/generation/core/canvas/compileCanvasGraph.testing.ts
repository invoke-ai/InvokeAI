/** Canvas compile fixtures, shared by the compiler's tests and the image recall contract. */

import type { GenerationProjectSettings } from '@features/generation/core/contracts';
import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateSettings,
  MainModelConfig,
  VaeModelConfig,
} from '@features/generation/core/types';

import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { getFamilyGraphBuilder } from '@features/generation/core/graph';

import type { ControlLayerGraphInput } from './addControlLayers';
import type { RegionalGuidanceInput } from './addRegionalGuidance';
import type { CanvasCompileMode, CanvasCompositingSettings, CanvasScalingSettings, Rect } from './types';

import { compileCanvasGraph } from './compileCanvasGraph';

const fluxVae: VaeModelConfig = { base: 'flux', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' };
export const qwenImageVae: VaeModelConfig = { base: 'qwen-image', key: 'qwen-vae', name: 'Qwen VAE', type: 'vae' };
const t5Encoder: ComponentModelConfig = { base: 'any', key: 't5', name: 'T5 Encoder', type: 't5_encoder' };
const clipEmbed: ComponentModelConfig = {
  base: 'any',
  key: 'clip',
  name: 'CLIP Embed',
  type: 'clip_embed',
  variant: 'large',
};
export const qwen3Encoder: ComponentModelConfig = {
  base: 'any',
  key: 'qwen3',
  name: 'Qwen3 Encoder',
  type: 'qwen3_encoder',
  variant: 'qwen3_06b',
};

export const PROJECT_SETTINGS: GenerationProjectSettings = {
  useCpuNoise: true,
  useNsfwChecker: false,
  useWatermark: false,
};

export const settingsFor = (
  model: GenerateModelConfig,
  overrides: Partial<GenerateSettings> = {}
): GenerateSettings => ({
  ...getDefaultGenerateSettings(model),
  seed: 1,
  seedMode: 'fixed',
  ...overrides,
});

/** Per-base overrides needed to pass component validation. */
export const COMPONENT_OVERRIDES: Partial<Record<string, Partial<GenerateSettings>>> = {
  flux: { clipEmbedModel: clipEmbed, t5EncoderModel: t5Encoder, vae: fluxVae },
  anima: { qwen3EncoderModel: qwen3Encoder, vae: qwenImageVae },
};

export const bbox: Rect = { height: 1024, width: 768, x: 128, y: 64 };

export const COMPOSITING_FIXTURE: CanvasCompositingSettings = {
  coherenceEdgeSize: 16,
  coherenceMinDenoise: 0,
  coherenceMode: 'Gaussian Blur',
  infillColorValue: { a: 1, b: 0, g: 0, r: 0 },
  infillMethod: 'lama',
  infillPatchmatchDownscaleSize: 1,
  infillTileSize: 32,
  maskBlur: 16,
  outputOnlyMaskedRegions: false,
};

/** Compiles `mode` over the fixture frame; settings that name every key replace the defaults and overrides. */
export const compile = (
  model: GenerateModelConfig,
  mode: CanvasCompileMode,
  overrides: {
    settings?: Partial<GenerateSettings>;
    bbox?: Rect;
    canvasMetadata?: Record<string, unknown>;
    compositeImageName?: string | null;
    maskImageName?: string | null;
    noiseMaskImageName?: string | null;
    compositing?: Partial<CanvasCompositingSettings>;
    scaling?: CanvasScalingSettings;
    strength?: number;
    destination?: 'canvas' | 'gallery';
    outputOnlyMaskedRegions?: boolean;
    controlLayers?: readonly ControlLayerGraphInput[];
    regionalGuidance?: readonly RegionalGuidanceInput[];
    projectSettings?: Partial<GenerationProjectSettings>;
  } = {}
) =>
  compileCanvasGraph({
    bbox: overrides.bbox ?? bbox,
    canvasMetadata: overrides.canvasMetadata,
    compositeImageName:
      'compositeImageName' in overrides ? (overrides.compositeImageName ?? null) : 'canvas-composite.png',
    compositing: {
      ...COMPOSITING_FIXTURE,
      outputOnlyMaskedRegions: overrides.outputOnlyMaskedRegions ?? false,
      ...overrides.compositing,
    },
    controlLayers: overrides.controlLayers,
    regionalGuidance: overrides.regionalGuidance,
    scaling: overrides.scaling,
    destination: overrides.destination ?? 'canvas',
    maskImageName:
      'maskImageName' in overrides ? overrides.maskImageName : mode === 'inpaint' ? 'canvas-mask.png' : null,
    mode,
    model,
    noiseMaskImageName: overrides.noiseMaskImageName ?? null,
    projectSettings: { ...PROJECT_SETTINGS, ...overrides.projectSettings },
    settings: settingsFor(model, {
      ...COMPONENT_OVERRIDES[model.base],
      ...overrides.settings,
    }),
    strength: overrides.strength ?? 0.6,
  });

/** The modes a local model compiles on the canvas: the image modes encode through the family's VAE, if it has one. */
export const canvasModesFor = (model: MainModelConfig, settings: GenerateSettings): CanvasCompileMode[] =>
  getFamilyGraphBuilder(model.base)?.(settings, model, false, PROJECT_SETTINGS).latents
    ? ['txt2img', 'img2img', 'inpaint']
    : ['txt2img'];
