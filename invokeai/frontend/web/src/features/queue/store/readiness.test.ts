import { describe, expect, it, vi } from 'vitest';

vi.mock('features/dynamicPrompts/util/getShouldProcessPrompt', () => ({
  getShouldProcessPrompt: vi.fn(() => false),
}));

vi.mock('i18next', () => ({
  default: {
    t: (key: string) => key,
  },
}));

import type { ParamsState, RefImagesState } from 'features/controlLayers/store/types';
import type { DynamicPromptsState } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import type { MainModelConfig, MainOrExternalModelConfig } from 'services/api/types';

import { getReasonsWhyCannotEnqueueCanvasTab, getReasonsWhyCannotEnqueueGenerateTab } from './readiness';

// --- Fixtures ---

const flux2DiffusersModel = {
  key: 'flux2-diff',
  hash: 'h',
  name: 'FLUX.2 Klein 4B',
  base: 'flux2',
  type: 'main',
  format: 'diffusers',
  variant: 'klein_4b',
} as unknown as MainModelConfig;

const flux2GGUF4BModel = {
  key: 'flux2-gguf-4b',
  hash: 'h',
  name: 'FLUX.2 Klein 4B GGUF',
  base: 'flux2',
  type: 'main',
  format: 'gguf_quantized',
  variant: 'klein_4b',
} as unknown as MainModelConfig;

const flux2GGUF9BModel = {
  key: 'flux2-gguf-9b',
  hash: 'h',
  name: 'FLUX.2 Klein 9B GGUF',
  base: 'flux2',
  type: 'main',
  format: 'gguf_quantized',
  variant: 'klein_9b',
} as unknown as MainModelConfig;

const kleinVaeModel = { key: 'vae', name: 'VAE', base: 'flux2', type: 'vae' };
const kleinQwen3Model = { key: 'qwen3', name: 'Qwen3', base: 'flux2', type: 'qwen3_encoder' };
const externalModel = {
  key: 'external',
  hash: 'h',
  name: 'External',
  base: 'external',
  type: 'external_image_generator',
  format: 'external_api',
} as unknown as MainOrExternalModelConfig;

const sdxlModel = {
  key: 'sdxl',
  hash: 'h',
  name: 'SDXL',
  base: 'sdxl',
  type: 'main',
  format: 'checkpoint',
} as unknown as MainModelConfig;

const upscaleModel = {
  key: 'upscale',
  hash: 'h',
  name: 'Upscale',
  base: 'any',
  type: 'spandrel_image_to_image',
};

const tileControlNetModel = {
  key: 'tile',
  hash: 'h',
  name: 'Tile ControlNet',
  base: 'sdxl',
  type: 'controlnet',
};

const baseDynamicPrompts: DynamicPromptsState = {
  _version: 1,
  maxPrompts: 100,
  combinatorial: false,
  prompts: ['test prompt'],
  parsingError: undefined,
  isError: false,
  isLoading: false,
  seedBehaviour: 'PER_PROMPT',
};

const baseRefImages: RefImagesState = {
  entities: [],
  ipAdapters: { entities: [], ids: [] },
} as unknown as RefImagesState;

const baseParams = {
  positivePrompt: 'test',
  kleinVaeModel: null,
  kleinQwen3EncoderModel: null,
  hrfEnabled: false,
  hrfMethod: 'latent',
  hrfUpscaleModel: null,
  hrfTileControlNetModel: null,
  hrfModel: null,
  hrfLoraMode: 'reuse_generate',
  hrfLoras: [],
  refinerModel: null,
} as unknown as ParamsState;

// --- Helpers ---

const buildGenerateTabArg = (overrides: {
  model?: MainOrExternalModelConfig | null;
  kleinVaeModel?: unknown;
  kleinQwen3EncoderModel?: unknown;
  hrfEnabled?: boolean;
  hrfMethod?: ParamsState['hrfMethod'];
  hrfUpscaleModel?: unknown;
  hrfTileControlNetModel?: unknown;
  hrfModel?: unknown;
  refinerModel?: unknown;
  hasFlux2DiffusersVaeSource?: boolean;
  hasFlux2DiffusersQwen3Source?: boolean;
}) => ({
  isConnected: true,
  model: overrides.model ?? flux2DiffusersModel,
  params: {
    ...baseParams,
    kleinVaeModel: overrides.kleinVaeModel ?? null,
    kleinQwen3EncoderModel: overrides.kleinQwen3EncoderModel ?? null,
    hrfEnabled: overrides.hrfEnabled ?? false,
    hrfMethod: overrides.hrfMethod ?? 'latent',
    hrfUpscaleModel: overrides.hrfUpscaleModel ?? null,
    hrfTileControlNetModel: overrides.hrfTileControlNetModel ?? null,
    hrfModel: overrides.hrfModel ?? null,
    refinerModel: overrides.refinerModel ?? null,
  } as unknown as ParamsState,
  refImages: baseRefImages,
  loras: [],
  dynamicPrompts: baseDynamicPrompts,
  hasFlux2DiffusersVaeSource: overrides.hasFlux2DiffusersVaeSource ?? false,
  hasFlux2DiffusersQwen3Source: overrides.hasFlux2DiffusersQwen3Source ?? false,
});

const buildCanvasTabArg = (overrides: {
  model?: MainModelConfig | null;
  kleinVaeModel?: unknown;
  kleinQwen3EncoderModel?: unknown;
  hasFlux2DiffusersVaeSource?: boolean;
  hasFlux2DiffusersQwen3Source?: boolean;
}) => ({
  isConnected: true,
  model: overrides.model ?? flux2DiffusersModel,
  canvas: {
    bbox: {
      scaleMethod: 'none',
      rect: { width: 1024, height: 1024 },
      scaledSize: { width: 1024, height: 1024 },
    },
    controlLayers: { entities: [] },
    regionalGuidance: { entities: [] },
    rasterLayers: { entities: [] },
    inpaintMasks: { entities: [] },
  },
  params: {
    ...baseParams,
    kleinVaeModel: overrides.kleinVaeModel ?? null,
    kleinQwen3EncoderModel: overrides.kleinQwen3EncoderModel ?? null,
  } as unknown as ParamsState,
  refImages: baseRefImages,
  loras: [],
  dynamicPrompts: baseDynamicPrompts,
  canvasIsFiltering: false,
  canvasIsTransforming: false,
  canvasIsRasterizing: false,
  canvasIsCompositing: false,
  canvasIsSelectingObject: false,
  hasFlux2DiffusersVaeSource: overrides.hasFlux2DiffusersVaeSource ?? false,
  hasFlux2DiffusersQwen3Source: overrides.hasFlux2DiffusersQwen3Source ?? false,
});

const hasFlux2VaeReason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('noFlux2KleinVaeModelSelected'));

const hasFlux2Qwen3Reason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('noFlux2KleinQwen3EncoderModelSelected'));

const hasHrfExternalReason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('hrfExternalModelUnsupported'));

const hasHrfRefinerReason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('hrfRefinerUnsupported'));

const hasHrfUpscaleModelBaseReason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('hrfUpscaleModelBaseUnsupported'));

const hasHrfUpscaleModelMissingReason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('hrfUpscaleModelMissing'));

const hasHrfTileControlNetMissingReason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('hrfTileControlNetModelMissing'));

const hasHrfModelOverrideBaseMismatchReason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('hrfModelOverrideBaseMismatch'));

// --- Tests ---

describe('FLUX.2 Klein readiness checks – generate tab', () => {
  it('no errors when main model is diffusers (VAE/Qwen3 extracted from it)', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildGenerateTabArg({ model: flux2DiffusersModel }));
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(false);
  });

  it('no errors when GGUF model with both VAE and Qwen3 diffusers sources', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({
        model: flux2GGUF4BModel,
        hasFlux2DiffusersVaeSource: true,
        hasFlux2DiffusersQwen3Source: true,
      })
    );
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(false);
  });

  it('errors for both VAE and Qwen3 when GGUF model with no diffusers source and no standalone models', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildGenerateTabArg({ model: flux2GGUF4BModel }));
    expect(hasFlux2VaeReason(reasons)).toBe(true);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(true);
  });

  it('errors only for Qwen3 when GGUF model with standalone VAE but no Qwen3 and no diffusers source', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({ model: flux2GGUF4BModel, kleinVaeModel: kleinVaeModel })
    );
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(true);
  });

  it('errors only for VAE when GGUF model with standalone Qwen3 but no VAE and no diffusers source', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({ model: flux2GGUF4BModel, kleinQwen3EncoderModel: kleinQwen3Model })
    );
    expect(hasFlux2VaeReason(reasons)).toBe(true);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(false);
  });

  it('no errors when GGUF model with both standalone VAE and Qwen3', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({
        model: flux2GGUF4BModel,
        kleinVaeModel: kleinVaeModel,
        kleinQwen3EncoderModel: kleinQwen3Model,
      })
    );
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(false);
  });

  it('VAE ok but Qwen3 errors when GGUF 9B model with only a 4B diffusers source (variant mismatch)', () => {
    // User has Klein 9B GGUF selected, only a 4B diffusers model installed.
    // VAE is shared across variants so it's ok. Qwen3 encoder differs, so it's not ok.
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({
        model: flux2GGUF9BModel,
        hasFlux2DiffusersVaeSource: true,
        hasFlux2DiffusersQwen3Source: false,
      })
    );
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(true);
  });

  it('no errors when GGUF 9B model with variant-matching diffusers source', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({
        model: flux2GGUF9BModel,
        hasFlux2DiffusersVaeSource: true,
        hasFlux2DiffusersQwen3Source: true,
      })
    );
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(false);
  });
});

describe('High Resolution Fix readiness checks - generate tab', () => {
  it('ignores stale HRF state for external models', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({ model: externalModel, hrfEnabled: true })
    );
    expect(hasHrfExternalReason(reasons)).toBe(false);
  });

  it('errors when HRF is enabled with SDXL Refiner', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({ model: sdxlModel, hrfEnabled: true, refinerModel: { key: 'refiner' } })
    );
    expect(hasHrfRefinerReason(reasons)).toBe(true);
  });

  it('ignores stale HRF state for unsupported model bases', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({
        model: flux2DiffusersModel,
        hrfEnabled: true,
        hrfMethod: 'upscale_model',
        hrfUpscaleModel: null,
        hrfTileControlNetModel: null,
        hrfModel: { key: 'anima', hash: 'h', name: 'Anima', base: 'anima', type: 'main' },
      })
    );
    expect(hasHrfUpscaleModelBaseReason(reasons)).toBe(false);
    expect(hasHrfUpscaleModelMissingReason(reasons)).toBe(false);
    expect(hasHrfTileControlNetMissingReason(reasons)).toBe(false);
    expect(hasHrfModelOverrideBaseMismatchReason(reasons)).toBe(false);
  });

  it('errors when upscale-model HRF is missing required models', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({ model: sdxlModel, hrfEnabled: true, hrfMethod: 'upscale_model' })
    );
    expect(hasHrfUpscaleModelMissingReason(reasons)).toBe(true);
    expect(hasHrfTileControlNetMissingReason(reasons)).toBe(true);
  });

  it('does not error when upscale-model HRF has required SDXL models', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({
        model: sdxlModel,
        hrfEnabled: true,
        hrfMethod: 'upscale_model',
        hrfUpscaleModel: upscaleModel,
        hrfTileControlNetModel: tileControlNetModel,
      })
    );
    expect(hasHrfUpscaleModelBaseReason(reasons)).toBe(false);
    expect(hasHrfUpscaleModelMissingReason(reasons)).toBe(false);
    expect(hasHrfTileControlNetMissingReason(reasons)).toBe(false);
  });

  it('does not apply stale upscale-model-only readiness checks to latent HRF', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({
        model: sdxlModel,
        hrfEnabled: true,
        hrfMethod: 'latent',
        hrfModel: { key: 'sd1', hash: 'h', name: 'SD1', base: 'sd-1', type: 'main' },
        hrfUpscaleModel: null,
        hrfTileControlNetModel: null,
      })
    );

    expect(hasHrfUpscaleModelMissingReason(reasons)).toBe(false);
    expect(hasHrfTileControlNetMissingReason(reasons)).toBe(false);
    expect(hasHrfModelOverrideBaseMismatchReason(reasons)).toBe(false);
  });

  it('errors when dedicated HRF model base differs from the Generate model base', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({
        model: sdxlModel,
        hrfEnabled: true,
        hrfMethod: 'upscale_model',
        hrfUpscaleModel: upscaleModel,
        hrfTileControlNetModel: tileControlNetModel,
        hrfModel: { key: 'sd1', hash: 'h', name: 'SD1', base: 'sd-1', type: 'main' },
      })
    );
    expect(hasHrfModelOverrideBaseMismatchReason(reasons)).toBe(true);
  });

  it('validates Tile ControlNet against the dedicated HRF model base', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildGenerateTabArg({
        model: sdxlModel,
        hrfEnabled: true,
        hrfMethod: 'upscale_model',
        hrfUpscaleModel: upscaleModel,
        hrfTileControlNetModel: { ...tileControlNetModel, base: 'sd-1' },
        hrfModel: { key: 'hrf-sdxl', hash: 'h', name: 'HRF SDXL', base: 'sdxl', type: 'main' },
      })
    );
    expect(hasHrfTileControlNetMissingReason(reasons)).toBe(true);
  });
});

describe('FLUX.2 Klein readiness checks – canvas tab', () => {
  it('no errors when main model is diffusers', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(buildCanvasTabArg({ model: flux2DiffusersModel }) as never);
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(false);
  });

  it('no errors when GGUF model with both VAE and Qwen3 diffusers sources', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildCanvasTabArg({
        model: flux2GGUF4BModel,
        hasFlux2DiffusersVaeSource: true,
        hasFlux2DiffusersQwen3Source: true,
      }) as never
    );
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(false);
  });

  it('errors for both VAE and Qwen3 when GGUF model with no sources', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(buildCanvasTabArg({ model: flux2GGUF4BModel }) as never);
    expect(hasFlux2VaeReason(reasons)).toBe(true);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(true);
  });

  it('no errors when GGUF model with both standalone VAE and Qwen3', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildCanvasTabArg({
        model: flux2GGUF4BModel,
        kleinVaeModel: kleinVaeModel,
        kleinQwen3EncoderModel: kleinQwen3Model,
      }) as never
    );
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(false);
  });

  it('VAE ok but Qwen3 errors when GGUF 9B with variant-mismatched diffusers source', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildCanvasTabArg({
        model: flux2GGUF9BModel,
        hasFlux2DiffusersVaeSource: true,
        hasFlux2DiffusersQwen3Source: false,
      }) as never
    );
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(true);
  });
});
