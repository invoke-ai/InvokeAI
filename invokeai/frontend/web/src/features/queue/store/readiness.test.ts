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
import type { MainModelConfig } from 'services/api/types';

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

const flux2VaeModel = { key: 'vae', name: 'VAE', base: 'flux2', type: 'vae' };
const kleinQwen3Model = { key: 'qwen3', name: 'Qwen3', base: 'flux2', type: 'qwen3_encoder' };

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
  flux2VaeModel: null,
  kleinQwen3EncoderModel: null,
} as unknown as ParamsState;

// --- Helpers ---

const buildGenerateTabArg = (overrides: {
  model?: MainModelConfig | null;
  flux2VaeModel?: unknown;
  kleinQwen3EncoderModel?: unknown;
  hasFlux2DiffusersVaeSource?: boolean;
  hasFlux2DiffusersQwen3Source?: boolean;
  hasFlux2DevDiffusersSource?: boolean;
}) => ({
  isConnected: true,
  model: overrides.model ?? flux2DiffusersModel,
  params: {
    ...baseParams,
    flux2VaeModel: overrides.flux2VaeModel ?? null,
    kleinQwen3EncoderModel: overrides.kleinQwen3EncoderModel ?? null,
  } as unknown as ParamsState,
  refImages: baseRefImages,
  loras: [],
  dynamicPrompts: baseDynamicPrompts,
  hasFlux2DiffusersVaeSource: overrides.hasFlux2DiffusersVaeSource ?? false,
  hasFlux2DiffusersQwen3Source: overrides.hasFlux2DiffusersQwen3Source ?? false,
  hasFlux2DevDiffusersSource: overrides.hasFlux2DevDiffusersSource ?? false,
});

const buildCanvasTabArg = (overrides: {
  model?: MainModelConfig | null;
  flux2VaeModel?: unknown;
  kleinQwen3EncoderModel?: unknown;
  hasFlux2DiffusersVaeSource?: boolean;
  hasFlux2DiffusersQwen3Source?: boolean;
  hasFlux2DevDiffusersSource?: boolean;
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
    flux2VaeModel: overrides.flux2VaeModel ?? null,
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
  hasFlux2DevDiffusersSource: overrides.hasFlux2DevDiffusersSource ?? false,
});

const hasFlux2VaeReason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('noFlux2KleinVaeModelSelected'));

const hasFlux2Qwen3Reason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('noFlux2KleinQwen3EncoderModelSelected'));

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
      buildGenerateTabArg({ model: flux2GGUF4BModel, flux2VaeModel: flux2VaeModel })
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
        flux2VaeModel: flux2VaeModel,
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
        flux2VaeModel: flux2VaeModel,
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

const ideogram4Model = {
  key: 'ideogram-4',
  hash: 'h',
  name: 'Ideogram 4',
  base: 'ideogram-4',
  type: 'main',
  format: 'diffusers',
} as unknown as MainModelConfig;

const buildIdeogram4CanvasArg = (canvasOverrides: {
  bbox?: { width: number; height: number };
  rasterLayers?: unknown[];
  inpaintMasks?: unknown[];
  regionalGuidance?: unknown[];
}) => ({
  ...buildCanvasTabArg({ model: ideogram4Model }),
  canvas: {
    bbox: {
      scaleMethod: 'none',
      rect: canvasOverrides.bbox ?? { width: 1024, height: 1024 },
      scaledSize: canvasOverrides.bbox ?? { width: 1024, height: 1024 },
    },
    controlLayers: { entities: [] },
    regionalGuidance: { entities: canvasOverrides.regionalGuidance ?? [] },
    rasterLayers: { entities: canvasOverrides.rasterLayers ?? [] },
    inpaintMasks: { entities: canvasOverrides.inpaintMasks ?? [] },
  },
});

const hasReasonWith = (reasons: { content: string }[], key: string) => reasons.some((r) => r.content.includes(key));

describe('Ideogram 4 readiness checks - canvas tab', () => {
  it('blocks a bbox whose width is not a multiple of 16', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildIdeogram4CanvasArg({ bbox: { width: 1025, height: 1024 } }) as never
    );
    expect(hasReasonWith(reasons, 'modelIncompatibleBboxWidth')).toBe(true);
  });

  it('allows a bbox that is a multiple of 16', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildIdeogram4CanvasArg({ bbox: { width: 1024, height: 1024 } }) as never
    );
    expect(hasReasonWith(reasons, 'modelIncompatibleBbox')).toBe(false);
  });

  it('blocks an enabled raster layer with content (Ideogram 4 is txt2img only)', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildIdeogram4CanvasArg({
        rasterLayers: [{ id: 'r1', isEnabled: true, type: 'raster_layer', objects: [{}] }],
      }) as never
    );
    expect(hasReasonWith(reasons, 'ideogram4Txt2ImgOnly')).toBe(true);
  });

  it('blocks an enabled inpaint mask with content', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildIdeogram4CanvasArg({
        inpaintMasks: [{ id: 'm1', isEnabled: true, type: 'inpaint_mask', objects: [{}] }],
      }) as never
    );
    expect(hasReasonWith(reasons, 'ideogram4Txt2ImgOnly')).toBe(true);
  });

  it('does not block an empty (fully transparent) enabled raster layer', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildIdeogram4CanvasArg({
        rasterLayers: [{ id: 'r1', isEnabled: true, type: 'raster_layer', objects: [] }],
      }) as never
    );
    expect(hasReasonWith(reasons, 'ideogram4Txt2ImgOnly')).toBe(false);
  });

  it('warns a regional guidance layer whose only input is a negative prompt', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildIdeogram4CanvasArg({
        regionalGuidance: [
          {
            id: 'rg1',
            isEnabled: true,
            type: 'regional_guidance',
            objects: [{}],
            positivePrompt: null,
            negativePrompt: 'no cats',
            autoNegative: false,
            referenceImages: [],
          },
        ],
      }) as never
    );
    expect(hasReasonWith(reasons, 'rgNegativePromptNotSupported')).toBe(true);
  });

  it('warns a regional guidance layer whose only input is a reference image', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildIdeogram4CanvasArg({
        regionalGuidance: [
          {
            id: 'rg1',
            isEnabled: true,
            type: 'regional_guidance',
            objects: [{}],
            positivePrompt: null,
            negativePrompt: null,
            autoNegative: false,
            referenceImages: [{ id: 'ri1', config: { model: null, image: null } }],
          },
        ],
      }) as never
    );
    expect(hasReasonWith(reasons, 'rgReferenceImagesNotSupported')).toBe(true);
  });
});
