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
import type { AnyModelConfig, MainModelConfig } from 'services/api/types';

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

const flux2SdnqPipelineModel = {
  key: 'flux2-sdnq',
  hash: 'h',
  name: 'FLUX.2 Klein 4B SDNQ',
  base: 'flux2',
  type: 'main',
  format: 'sdnq_quantized',
  variant: 'klein_4b',
  submodels: { transformer: {}, vae: {}, text_encoder: {}, tokenizer: {} },
} as unknown as MainModelConfig;

const flux2SdnqPartialModel = {
  ...flux2SdnqPipelineModel,
  key: 'flux2-sdnq-partial',
  submodels: { transformer: {} },
} as unknown as MainModelConfig;

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
  wanWiredConfigs: { vae: null, componentSource: null, lowNoisePartner: null },
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
  wanWiredConfigs: { vae: null, componentSource: null, lowNoisePartner: null },
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

describe('FLUX.2 Klein SDNQ pipeline readiness checks', () => {
  it('generate: no errors for a full SDNQ pipeline (self-contained) with no component sources', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildGenerateTabArg({ model: flux2SdnqPipelineModel }));
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(false);
  });

  it('generate: errors for a partial SDNQ pipeline (only transformer submodel) with no sources', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildGenerateTabArg({ model: flux2SdnqPartialModel }));
    expect(hasFlux2VaeReason(reasons)).toBe(true);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(true);
  });

  it('canvas: no errors for a full SDNQ pipeline (self-contained) with no component sources', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(buildCanvasTabArg({ model: flux2SdnqPipelineModel }) as never);
    expect(hasFlux2VaeReason(reasons)).toBe(false);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(false);
  });

  it('canvas: errors for a partial SDNQ pipeline (only transformer submodel) with no sources', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(buildCanvasTabArg({ model: flux2SdnqPartialModel }) as never);
    expect(hasFlux2VaeReason(reasons)).toBe(true);
    expect(hasFlux2Qwen3Reason(reasons)).toBe(true);
  });
});

const zImageSdnqPipelineModel = {
  key: 'z-image-sdnq',
  hash: 'h',
  name: 'Z-Image Turbo SDNQ',
  base: 'z-image',
  type: 'main',
  format: 'sdnq_quantized',
  submodels: { transformer: {}, vae: {}, text_encoder: {}, tokenizer: {} },
} as unknown as MainModelConfig;

const zImageGgufModel = {
  key: 'z-image-gguf',
  hash: 'h',
  name: 'Z-Image Turbo GGUF',
  base: 'z-image',
  type: 'main',
  format: 'gguf_quantized',
} as unknown as MainModelConfig;

const buildZImageTabArg = (overrides: {
  model?: MainModelConfig | null;
  zImageVaeModel?: unknown;
  zImageQwen3EncoderModel?: unknown;
  zImageQwen3SourceModel?: unknown;
}) => ({
  isConnected: true,
  model: overrides.model ?? zImageGgufModel,
  params: {
    ...baseParams,
    zImageVaeModel: overrides.zImageVaeModel ?? null,
    zImageQwen3EncoderModel: overrides.zImageQwen3EncoderModel ?? null,
    zImageQwen3SourceModel: overrides.zImageQwen3SourceModel ?? null,
  } as unknown as ParamsState,
  refImages: baseRefImages,
  loras: [],
  dynamicPrompts: baseDynamicPrompts,
  hasFlux2DiffusersVaeSource: false,
  hasFlux2DiffusersQwen3Source: false,
  hasFlux2DevDiffusersSource: false,
  wanWiredConfigs: { vae: null, componentSource: null, lowNoisePartner: null },
});

const hasZImageVaeReason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('noZImageVaeSourceSelected'));

const hasZImageQwen3Reason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('noZImageQwen3EncoderSourceSelected'));

describe('Z-Image readiness checks – generate tab', () => {
  it('no errors when main model is a self-contained SDNQ pipeline (no component source selected)', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildZImageTabArg({ model: zImageSdnqPipelineModel }));
    expect(hasZImageVaeReason(reasons)).toBe(false);
    expect(hasZImageQwen3Reason(reasons)).toBe(false);
  });

  it('errors for both VAE and Qwen3 when GGUF model with no component source selected', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildZImageTabArg({ model: zImageGgufModel }));
    expect(hasZImageVaeReason(reasons)).toBe(true);
    expect(hasZImageQwen3Reason(reasons)).toBe(true);
  });

  it('no errors when GGUF model has a Qwen3 Source (supplies both VAE and encoder)', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildZImageTabArg({ model: zImageGgufModel, zImageQwen3SourceModel: { key: 'src' } })
    );
    expect(hasZImageVaeReason(reasons)).toBe(false);
    expect(hasZImageQwen3Reason(reasons)).toBe(false);
  });

  it('errors only for VAE when GGUF model has a standalone Qwen3 encoder but no VAE source', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildZImageTabArg({ model: zImageGgufModel, zImageQwen3EncoderModel: { key: 'enc' } })
    );
    expect(hasZImageVaeReason(reasons)).toBe(true);
    expect(hasZImageQwen3Reason(reasons)).toBe(false);
  });

  it('does not treat a non-pipeline SDNQ model (no submodels) as self-contained', () => {
    const zImageSdnqSingleFile = {
      ...zImageSdnqPipelineModel,
      submodels: undefined,
    } as unknown as MainModelConfig;
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildZImageTabArg({ model: zImageSdnqSingleFile }));
    expect(hasZImageVaeReason(reasons)).toBe(true);
    expect(hasZImageQwen3Reason(reasons)).toBe(true);
  });

  it('does not treat a partial SDNQ pipeline (missing vae/text_encoder/tokenizer) as self-contained', () => {
    const zImageSdnqPartial = {
      ...zImageSdnqPipelineModel,
      submodels: { transformer: {} },
    } as unknown as MainModelConfig;
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildZImageTabArg({ model: zImageSdnqPartial }));
    expect(hasZImageVaeReason(reasons)).toBe(true);
    expect(hasZImageQwen3Reason(reasons)).toBe(true);
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

// --- PiD Native-mode scaled-grid validation (SD3 / SDXL / Z-Image) ---
// In Native mode the Canvas bbox is the 4x target and is generated at bbox/4, so the bbox must be a
// multiple of grid*4 (SD3/Z-Image grid 16 -> 64, SDXL grid 8 -> 32). Without validation an off-grid bbox
// silently becomes a smaller generation. FLUX/FLUX.2/Qwen already validated this; these tests cover the
// three bases whose checks previously omitted it.

const sd3Model = {
  key: 'sd3',
  hash: 'h',
  name: 'SD3',
  base: 'sd-3',
  type: 'main',
  format: 'diffusers',
} as unknown as MainModelConfig;
const sdxlModel = {
  key: 'sdxl',
  hash: 'h',
  name: 'SDXL',
  base: 'sdxl',
  type: 'main',
  format: 'diffusers',
} as unknown as MainModelConfig;
const zImageModel = {
  key: 'zimg',
  hash: 'h',
  name: 'Z-Image',
  base: 'z-image',
  type: 'main',
  format: 'diffusers',
} as unknown as MainModelConfig;

const buildPidCanvasArg = (model: MainModelConfig, bboxSide: number) => ({
  isConnected: true,
  model,
  canvas: {
    bbox: {
      scaleMethod: 'none',
      rect: { width: bboxSide, height: bboxSide },
      scaledSize: { width: bboxSide, height: bboxSide },
    },
    controlLayers: { entities: [] },
    regionalGuidance: { entities: [] },
    rasterLayers: { entities: [] },
    inpaintMasks: { entities: [] },
  },
  params: {
    ...baseParams,
    // Satisfy the non-grid PiD requirements so only the grid reason (if any) remains under test.
    pidMode: 'native',
    pidDecoderModel: { key: 'dec', name: 'decoder', base: model.base === 'z-image' ? 'flux' : model.base },
    gemma2EncoderModel: { key: 'gem', name: 'gemma', base: 'any' },
    refinerModel: null,
    zImageVaeModel: { key: 'zvae', name: 'zvae', base: 'z-image', type: 'vae' },
    zImageQwen3SourceModel: { key: 'zsrc', name: 'zsrc', base: 'z-image', type: 'main' },
    zImageQwen3EncoderModel: null,
  } as unknown as ParamsState,
  refImages: baseRefImages,
  loras: [],
  dynamicPrompts: baseDynamicPrompts,
  canvasIsFiltering: false,
  canvasIsTransforming: false,
  canvasIsRasterizing: false,
  canvasIsCompositing: false,
  canvasIsSelectingObject: false,
  hasFlux2DiffusersVaeSource: false,
  hasFlux2DiffusersQwen3Source: false,
  hasFlux2DevDiffusersSource: false,
  wanWiredConfigs: { vae: null, componentSource: null, lowNoisePartner: null },
});

const hasBboxGridReason = (reasons: { content: string }[]) =>
  reasons.some(
    (r) => r.content.includes('modelIncompatibleBboxWidth') || r.content.includes('modelIncompatibleBboxHeight')
  );

describe('PiD Native scaled-grid readiness – canvas tab', () => {
  it.each([
    ['SD3', sd3Model],
    ['SDXL', sdxlModel],
    ['Z-Image', zImageModel],
  ] as const)('blocks an off-grid 1040px bbox in Native mode for %s', (_label, model) => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(buildPidCanvasArg(model, 1040) as never);
    expect(hasBboxGridReason(reasons)).toBe(true);
  });

  it.each([
    ['SD3', sd3Model],
    ['SDXL', sdxlModel],
    ['Z-Image', zImageModel],
  ] as const)('allows an on-grid 1024px bbox in Native mode for %s', (_label, model) => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(buildPidCanvasArg(model, 1024) as never);
    expect(hasBboxGridReason(reasons)).toBe(false);
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

// ---------------------------------------------------------------------------
// FLUX.1: a complete SDNQ pipeline supplies its own T5 / CLIP / VAE
// ---------------------------------------------------------------------------

// `submodels` is not on MainModelConfig (the generated schema doesn't carry the SDNQ variants), so
// keep the map as its own const rather than reading it back through the cast.
const flux1PipelineSubmodels = {
  transformer: {},
  vae: {},
  text_encoder: {},
  tokenizer: {},
  text_encoder_2: {},
  tokenizer_2: {},
};

const flux1SdnqPipelineModel = {
  key: 'flux1-sdnq-pipeline',
  hash: 'flux1-sdnq-hash',
  name: 'FLUX.1 dev SDNQ',
  base: 'flux',
  type: 'main',
  format: 'sdnq_quantized',
  variant: 'dev',
  submodels: flux1PipelineSubmodels,
} as unknown as MainModelConfig;

const flux1SingleFileModel = {
  key: 'flux1-gguf',
  hash: 'flux1-gguf-hash',
  name: 'FLUX.1 dev GGUF',
  base: 'flux',
  type: 'main',
  format: 'gguf_quantized',
  variant: 'dev',
} as unknown as MainModelConfig;

const flux1ComponentReasons = (reasons: { content: string }[]) =>
  reasons.filter(
    (r) =>
      r.content.includes('noT5EncoderModelSelected') ||
      r.content.includes('noCLIPEmbedModelSelected') ||
      r.content.includes('noFLUXVAEModelSelected')
  );

describe('FLUX.1 readiness – self-contained SDNQ pipeline', () => {
  it('does not demand standalone T5 / CLIP / VAE when the pipeline ships them', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildGenerateTabArg({ model: flux1SdnqPipelineModel }));

    expect(flux1ComponentReasons(reasons)).toEqual([]);
  });

  it('still demands all three for a single-file FLUX.1 model', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildGenerateTabArg({ model: flux1SingleFileModel }));

    expect(flux1ComponentReasons(reasons)).toHaveLength(3);
  });

  it('still demands them when the pipeline is missing its T5 pair', () => {
    const { text_encoder_2: _te2, tokenizer_2: _tok2, ...withoutT5 } = flux1PipelineSubmodels;
    const partial = { ...flux1SdnqPipelineModel, submodels: withoutT5 } as unknown as MainModelConfig;

    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildGenerateTabArg({ model: partial }));

    expect(flux1ComponentReasons(reasons)).toHaveLength(3);
  });
});

// --- Wan 2.2 -----------------------------------------------------------------
//
// Regression cover for #9463: single-file Wan mains are transformer-only and need a
// VAE + UMT5-XXL encoder from elsewhere. That was originally gated on the GGUF
// format alone, so when the safetensors checkpoint format was added the pre-flight
// silently skipped it and Invoke was enabled for a graph that could only fail in
// the model loader.

const wanGgufModel = {
  key: 'wan-gguf',
  hash: 'h',
  name: 'Wan 2.2 T2V A14B GGUF',
  base: 'wan',
  type: 'main',
  format: 'gguf_quantized',
  variant: 't2v_a14b',
  expert: 'high',
} as unknown as MainModelConfig;

const wanCheckpointModel = {
  key: 'wan-checkpoint',
  hash: 'h',
  name: 'Wan 2.2 T2V A14B safetensors',
  base: 'wan',
  type: 'main',
  format: 'checkpoint',
  variant: 't2v_a14b',
  expert: 'high',
} as unknown as MainModelConfig;

const wanDiffusersModel = {
  key: 'wan-diffusers',
  hash: 'h',
  name: 'Wan 2.2 T2V A14B Diffusers',
  base: 'wan',
  type: 'main',
  format: 'diffusers',
  variant: 't2v_a14b',
} as unknown as MainModelConfig;

/** A14B checkpoint whose filename carried no high/low marker, so the probe recorded
 *  expert='none'. Extremely common on community finetunes — the tag is a filename
 *  heuristic and there is no UI to correct it. */
const wanUntaggedA14bModel = {
  key: 'wan-untagged',
  hash: 'h',
  name: 'wan2.2_t2v_A14B_fp8_e4m3fn',
  base: 'wan',
  type: 'main',
  format: 'checkpoint',
  variant: 't2v_a14b',
  expert: 'none',
} as unknown as MainModelConfig;

const wanLowExpertModel = {
  ...wanUntaggedA14bModel,
  key: 'wan-low',
  name: 'Wan2.2-T2V-A14B-LOW',
  expert: 'low',
} as unknown as MainModelConfig;

/** TI2V-5B is single-transformer, so the A14B expert pairing does not apply to it. */
const wanTi2v5bModel = {
  key: 'wan-5b',
  hash: 'h',
  name: 'Wan2.2 TI2V 5B',
  base: 'wan',
  type: 'main',
  format: 'checkpoint',
  variant: 'ti2v_5b',
  expert: 'none',
} as unknown as MainModelConfig;

const buildWanTabArg = (overrides: {
  model?: MainModelConfig | null;
  wanVaeModel?: unknown;
  wanT5EncoderModel?: unknown;
  wanComponentSource?: unknown;
  wanTransformerLowNoise?: unknown;
  wiredVae?: unknown;
  wiredComponentSource?: unknown;
  wiredLowNoisePartner?: unknown;
}) => ({
  isConnected: true,
  model: overrides.model ?? wanCheckpointModel,
  params: {
    ...baseParams,
    wanVaeModel: overrides.wanVaeModel ?? null,
    wanT5EncoderModel: overrides.wanT5EncoderModel ?? null,
    wanComponentSource: overrides.wanComponentSource ?? null,
    wanTransformerLowNoise: overrides.wanTransformerLowNoise ?? null,
    model: overrides.model ?? wanCheckpointModel,
  } as unknown as ParamsState,
  refImages: baseRefImages,
  loras: [],
  dynamicPrompts: baseDynamicPrompts,
  hasFlux2DiffusersVaeSource: false,
  hasFlux2DiffusersQwen3Source: false,
  hasFlux2DevDiffusersSource: false,
  wanWiredConfigs: {
    vae: (overrides.wiredVae ?? null) as AnyModelConfig | null,
    componentSource: (overrides.wiredComponentSource ?? null) as AnyModelConfig | null,
    lowNoisePartner: (overrides.wiredLowNoisePartner ?? null) as AnyModelConfig | null,
  },
});

const buildWanCanvasArg = (overrides: Parameters<typeof buildWanTabArg>[0]) =>
  ({
    ...buildWanTabArg(overrides),
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
    canvasIsFiltering: false,
    canvasIsTransforming: false,
    canvasIsRasterizing: false,
    canvasIsCompositing: false,
    canvasIsSelectingObject: false,
  }) as never;

const hasWanComponentReason = (reasons: { content: string }[]) =>
  reasons.some((r) => r.content.includes('noWanComponentSourceSelected'));

// Since #9505 the loader takes the A14B expert pairing from the wiring rather than the
// filename tag: an unpaired or untagged A14B runs with a warning instead of raising. The
// pre-flight must therefore NOT block on `expert`, or it would stop a generation the
// backend is happy to run — untagged community checkpoints are the common case this
// whole branch exists to support.
const vae16 = { key: 'vae16', name: 'Wan 2.1 VAE', base: 'wan', type: 'vae', latent_channels: 16 };
const vae48 = { key: 'vae48', name: 'Wan 2.2 VAE', base: 'wan', type: 'vae', latent_channels: 48 };
const a14bDiffusers = {
  key: 'a14b-diff',
  name: 'A14B Diffusers',
  base: 'wan',
  type: 'main',
  format: 'diffusers',
  variant: 't2v_a14b',
};
const ti2vDiffusers = {
  key: 'ti2v-diff',
  name: 'TI2V Diffusers',
  base: 'wan',
  type: 'main',
  format: 'diffusers',
  variant: 'ti2v_5b',
};

// The Advanced comboboxes offer every Wan VAE and every Wan Diffusers main with no
// variant filtering, and nothing re-runs the auto-fill when the user picks by hand — so
// presence alone is not enough. These are the loader's own checks, mirrored.
describe('Wan 2.2 component compatibility pre-flight', () => {
  const src = { wanComponentSource: { key: 'src' } };

  it('blocks a VAE whose channel count does not match the transformer', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildWanTabArg({ model: wanTi2v5bModel, ...src, wanVaeModel: vae16, wiredVae: vae16 })
    );
    expect(reasons.some((r) => r.content.includes('incompatibleWanVae'))).toBe(true);
  });

  it('accepts the matching VAE for each variant', () => {
    for (const [model, vae] of [
      [wanTi2v5bModel, vae48],
      [wanCheckpointModel, vae16],
    ] as const) {
      const reasons = getReasonsWhyCannotEnqueueGenerateTab(
        buildWanTabArg({ model, ...src, wanVaeModel: vae, wiredVae: vae })
      );
      expect(reasons.some((r) => r.content.includes('incompatibleWanVae'))).toBe(false);
    }
  });

  it('checks the VAE for a Diffusers main too — a wired standalone outranks its own', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildWanTabArg({ model: wanDiffusersModel, wanVaeModel: vae48, wiredVae: vae48 })
    );
    expect(reasons.some((r) => r.content.includes('incompatibleWanVae'))).toBe(true);
  });

  it('blocks a Component Source from the other variant family', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildWanTabArg({ model: wanTi2v5bModel, wanComponentSource: a14bDiffusers, wiredComponentSource: a14bDiffusers })
    );
    expect(reasons.some((r) => r.content.includes('incompatibleWanComponentSource'))).toBe(true);
  });

  it('accepts a same-family Component Source', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildWanTabArg({ model: wanTi2v5bModel, wanComponentSource: ti2vDiffusers, wiredComponentSource: ti2vDiffusers })
    );
    expect(reasons.some((r) => r.content.includes('incompatibleWanComponentSource'))).toBe(false);
  });

  it('blocks a low-noise partner of a different variant', () => {
    const i2vLow = { ...wanLowExpertModel, key: 'i2v-low', variant: 'i2v_a14b' };
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildWanTabArg({
        model: wanCheckpointModel,
        ...src,
        wanTransformerLowNoise: i2vLow,
        wiredLowNoisePartner: i2vLow,
      })
    );
    expect(reasons.some((r) => r.content.includes('incompatibleWanLowNoiseExpert'))).toBe(true);
  });

  it('blocks the same model wired to both transformer slots', () => {
    // Reachable whenever a low expert has no partner: it is offered by the main picker
    // (nothing else to offer) and by the low-noise picker at the same time.
    const params = { key: wanLowExpertModel.key };
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildWanTabArg({
        model: wanLowExpertModel,
        ...src,
        wanTransformerLowNoise: params,
        wiredLowNoisePartner: wanLowExpertModel,
      })
    );
    expect(reasons.some((r) => r.content.includes('duplicateWanTransformer'))).toBe(true);
  });

  it('does not judge a slot pointing at a deleted model as incompatible', () => {
    // wiredVae null means "empty or dangling"; neither can be checked for compatibility.
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildWanTabArg({ model: wanTi2v5bModel, ...src, wanVaeModel: { key: 'gone' }, wiredVae: null })
    );
    expect(reasons.some((r) => r.content.includes('incompatibleWanVae'))).toBe(false);
  });
});

describe('Wan 2.2 A14B expert pairing is not a readiness blocker', () => {
  const withComponents = { wanComponentSource: { key: 'src' } };

  it.each([
    ['untagged (expert=none)', wanUntaggedA14bModel],
    ['the low-noise expert', wanLowExpertModel],
    ['the high-noise expert', wanCheckpointModel],
    ['TI2V-5B', wanTi2v5bModel],
  ])('does not block an unpaired A14B main that is %s', (_label, model) => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildWanTabArg({ model, ...withComponents }));
    expect(reasons).toEqual([]);
  });

  it('also does not block on the canvas tab', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildWanCanvasArg({ model: wanUntaggedA14bModel, ...withComponents })
    );
    expect(reasons).toEqual([]);
  });

  it('still blocks when the VAE/encoder source is missing, whatever the expert', () => {
    // The component-source rule is independent of the pairing and must survive.
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildWanTabArg({ model: wanUntaggedA14bModel }));
    expect(hasWanComponentReason(reasons)).toBe(true);
  });
});

describe('Wan 2.2 readiness checks – generate tab', () => {
  it.each([
    ['GGUF', wanGgufModel],
    ['single-file checkpoint', wanCheckpointModel],
  ])('errors when a %s main has no VAE or encoder source', (_label, model) => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildWanTabArg({ model }));
    expect(hasWanComponentReason(reasons)).toBe(true);
  });

  it.each([
    ['GGUF', wanGgufModel],
    ['single-file checkpoint', wanCheckpointModel],
  ])('no error when a %s main has standalone VAE + encoder', (_label, model) => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildWanTabArg({ model, wanVaeModel: { key: 'vae' }, wanT5EncoderModel: { key: 't5' } })
    );
    expect(hasWanComponentReason(reasons)).toBe(false);
  });

  it('errors when only one of VAE / encoder is supplied', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildWanTabArg({ model: wanCheckpointModel, wanVaeModel: { key: 'vae' } })
    );
    expect(hasWanComponentReason(reasons)).toBe(true);
  });

  it('no error when a Component Source supplies both', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(
      buildWanTabArg({ model: wanCheckpointModel, wanComponentSource: { key: 'src' } })
    );
    expect(hasWanComponentReason(reasons)).toBe(false);
  });

  it('no error for a Diffusers main, which carries its own components', () => {
    const reasons = getReasonsWhyCannotEnqueueGenerateTab(buildWanTabArg({ model: wanDiffusersModel }));
    expect(hasWanComponentReason(reasons)).toBe(false);
  });
});

describe('Wan 2.2 readiness checks – canvas tab', () => {
  it.each([
    ['GGUF', wanGgufModel],
    ['single-file checkpoint', wanCheckpointModel],
  ])('errors when a %s main has no VAE or encoder source', (_label, model) => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(buildWanCanvasArg({ model }));
    expect(hasWanComponentReason(reasons)).toBe(true);
  });
});
