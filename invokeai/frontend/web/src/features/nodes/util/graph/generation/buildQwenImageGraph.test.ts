import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('app/logging/logger', () => ({
  logger: () => ({
    debug: vi.fn(),
  }),
}));

let nextId = 0;
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:${nextId++}`,
}));

const model = {
  key: 'qwen-model',
  hash: 'qwen-hash',
  name: 'Qwen Image Generate',
  base: 'qwen-image',
  type: 'main',
  variant: 'generate',
};

const defaultParams: {
  cfgScale: number | number[];
  steps: number;
  qwenImageComponentSource: null;
  qwenImageQuantization: string;
  qwenImageShift: number;
} = {
  cfgScale: 4,
  steps: 20,
  qwenImageComponentSource: null,
  qwenImageQuantization: 'none',
  qwenImageShift: 1,
};

let params = { ...defaultParams };

const refImagesSlice = {
  entities: [
    {
      id: 'ref-image-1',
      isEnabled: true,
      config: {
        type: 'qwen_image_reference_image',
        image: {
          original: {
            image: {
              image_name: 'reference.png',
              width: 512,
              height: 512,
            },
          },
        },
      },
    },
  ],
};

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn(() => model),
  selectParamsSlice: vi.fn(() => params),
}));

vi.mock('features/controlLayers/store/refImagesSlice', () => ({
  selectRefImagesSlice: vi.fn(() => refImagesSlice),
}));

vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasMetadata: vi.fn(() => ({})),
}));

vi.mock('features/controlLayers/store/types', () => ({
  isQwenImageReferenceImageConfig: vi.fn((config: { type?: string }) => config.type === 'qwen_image_reference_image'),
}));

vi.mock('features/controlLayers/store/validators', () => ({
  getGlobalReferenceImageWarnings: vi.fn(() => []),
}));

vi.mock('features/metadata/util/modelFetchingHelpers', () => ({
  fetchModelConfigWithTypeGuard: vi.fn(() => Promise.resolve(model)),
}));

vi.mock('features/nodes/types/common', async () => {
  const actual = await vi.importActual('features/nodes/types/common');
  return {
    ...actual,
    zImageField: {
      parse: vi.fn((image) => image),
    },
  };
});

vi.mock('features/nodes/util/graph/generation/addImageToImage', () => ({
  addImageToImage: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addInpaint', () => ({
  addInpaint: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addNSFWChecker', () => ({
  addNSFWChecker: vi.fn((_g, node) => node),
}));

vi.mock('features/nodes/util/graph/generation/addOutpaint', () => ({
  addOutpaint: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addQwenImageLoRAs', () => ({
  addQwenImageLoRAs: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addTextToImage', () => ({
  addTextToImage: vi.fn(({ l2i }) => l2i),
}));

vi.mock('features/nodes/util/graph/generation/addWatermarker', () => ({
  addWatermarker: vi.fn((_g, node) => node),
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  selectCanvasOutputFields: vi.fn(() => ({})),
  selectPresetModifiedPrompts: vi.fn(() => ({
    positive: 'a prompt',
    negative: 'a negative prompt',
  })),
}));

vi.mock('features/ui/store/uiSelectors', () => ({
  selectActiveTab: vi.fn(() => 'generation'),
}));

vi.mock('services/api/types', async () => {
  const actual = await vi.importActual('services/api/types');
  return {
    ...actual,
    isNonRefinerMainModelConfig: vi.fn(() => true),
  };
});

import { buildQwenImageGraph, isQwenImageEditModel, shouldUseCfg } from './buildQwenImageGraph';

describe('isQwenImageEditModel', () => {
  afterEach(() => {
    nextId = 0;
    params = { ...defaultParams };
  });

  it('returns true for edit variant', () => {
    expect(isQwenImageEditModel({ variant: 'edit' })).toBe(true);
  });

  it('returns false for generate variant', () => {
    expect(isQwenImageEditModel({ variant: 'generate' })).toBe(false);
  });

  it('returns false when variant is null', () => {
    expect(isQwenImageEditModel({ variant: null })).toBe(false);
  });

  it('returns false when variant is undefined', () => {
    expect(isQwenImageEditModel({ variant: undefined })).toBe(false);
  });

  it('returns false when variant field is absent', () => {
    expect(isQwenImageEditModel({})).toBe(false);
  });

  it('returns false when model is null', () => {
    expect(isQwenImageEditModel(null)).toBe(false);
  });

  it('returns false for unrelated variant values', () => {
    expect(isQwenImageEditModel({ variant: 'schnell' })).toBe(false);
    expect(isQwenImageEditModel({ variant: 'dev' })).toBe(false);
    expect(isQwenImageEditModel({ variant: 'turbo' })).toBe(false);
  });

  describe('reference image filtering regression', () => {
    it('prevents reference images from leaking to generate models when switching from edit', () => {
      const editModel = { variant: 'edit' as const };
      const generateModel = { variant: 'generate' as const };

      expect(isQwenImageEditModel(editModel)).toBe(true);
      expect(isQwenImageEditModel(generateModel)).toBe(false);
    });

    it('prevents reference images from leaking to GGUF models without variant', () => {
      const ggufModelNoVariant = {};
      expect(isQwenImageEditModel(ggufModelNoVariant)).toBe(false);
    });
  });
});

describe('shouldUseCfg', () => {
  afterEach(() => {
    nextId = 0;
    params = { ...defaultParams };
  });

  describe('negative conditioning is included when cfgScale > 1', () => {
    it('returns true for cfgScale = 4', () => {
      expect(shouldUseCfg(4)).toBe(true);
    });

    it('returns true for cfgScale = 1.5', () => {
      expect(shouldUseCfg(1.5)).toBe(true);
    });

    it('returns true for cfgScale = 1.01', () => {
      expect(shouldUseCfg(1.01)).toBe(true);
    });
  });

  describe('negative conditioning is excluded when cfgScale <= 1', () => {
    it('returns false for cfgScale = 1', () => {
      expect(shouldUseCfg(1)).toBe(false);
    });

    it('returns false for cfgScale = 0.5', () => {
      expect(shouldUseCfg(0.5)).toBe(false);
    });

    it('returns false for cfgScale = 0', () => {
      expect(shouldUseCfg(0)).toBe(false);
    });
  });

  describe('array cfgScale (per-step)', () => {
    it('returns true for per-step arrays with values > 1', () => {
      expect(shouldUseCfg([4, 3, 2, 1])).toBe(true);
    });

    it('returns true when any per-step cfg value is > 1', () => {
      expect(shouldUseCfg([1, 1.1, 1])).toBe(true);
      expect(shouldUseCfg([0.5, 2, 0.5])).toBe(true);
    });

    it('returns false when every per-step cfg value is <= 1', () => {
      expect(shouldUseCfg([1, 1, 1])).toBe(false);
      expect(shouldUseCfg([0.5, 0.75, 1])).toBe(false);
    });
  });

  describe('CFG gating regression', () => {
    it('with cfgScale=1, neg_prompt is absent from the graph (no wasted compute)', () => {
      expect(shouldUseCfg(1)).toBe(false);
    });

    it('with cfgScale=4, neg_prompt is present in the graph for classifier-free guidance', () => {
      expect(shouldUseCfg(4)).toBe(true);
    });

    it('omits negative conditioning edges from the graph when per-step cfg never exceeds 1', async () => {
      params = {
        ...defaultParams,
        cfgScale: [1, 1, 1],
      };

      const { g } = await buildQwenImageGraph({
        generationMode: 'txt2img',
        manager: null,
        state: {
          system: {
            shouldUseNSFWChecker: false,
            shouldUseWatermarker: false,
          },
        } as never,
      });

      const graph = g.getGraph();
      const nodeIds = Object.keys(graph.nodes);
      const hasNegativePromptNode = nodeIds.some((id) => id.startsWith('neg_prompt:'));
      const hasNegativeConditioningEdge = graph.edges.some(
        (edge) => edge.destination.field === 'negative_conditioning'
      );

      expect(hasNegativePromptNode).toBe(false);
      expect(hasNegativeConditioningEdge).toBe(false);
    });

    it('includes negative conditioning edges in the graph when any per-step cfg exceeds 1', async () => {
      params = {
        ...defaultParams,
        cfgScale: [1, 2, 1],
      };

      const { g } = await buildQwenImageGraph({
        generationMode: 'txt2img',
        manager: null,
        state: {
          system: {
            shouldUseNSFWChecker: false,
            shouldUseWatermarker: false,
          },
        } as never,
      });

      const graph = g.getGraph();
      const nodeIds = Object.keys(graph.nodes);
      const hasNegativePromptNode = nodeIds.some((id) => id.startsWith('neg_prompt:'));
      const hasNegativeConditioningEdge = graph.edges.some(
        (edge) => edge.destination.field === 'negative_conditioning'
      );

      expect(hasNegativePromptNode).toBe(true);
      expect(hasNegativeConditioningEdge).toBe(true);
    });
  });
});

describe('buildQwenImageGraph', () => {
  afterEach(() => {
    nextId = 0;
    params = { ...defaultParams };
  });

  it('does not include hidden Qwen reference images for generate-variant models', async () => {
    const { g } = await buildQwenImageGraph({
      generationMode: 'txt2img',
      manager: null,
      state: {
        system: {
          shouldUseNSFWChecker: false,
          shouldUseWatermarker: false,
        },
      } as never,
    });

    const graph = g.getGraph();
    const nodeIds = Object.keys(graph.nodes);
    const hasReferenceCollectionNode = nodeIds.some((id) => id.startsWith('qwen_ref_img_collect:'));
    const hasReferenceImagesEdge = graph.edges.some((edge) => edge.destination.field === 'reference_images');
    const hasReferenceLatentsEdge = graph.edges.some((edge) => edge.destination.field === 'reference_latents');

    expect(hasReferenceCollectionNode).toBe(false);
    expect(hasReferenceImagesEdge).toBe(false);
    expect(hasReferenceLatentsEdge).toBe(false);
  });
});
