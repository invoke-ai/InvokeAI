import { afterEach, describe, expect, it, vi } from 'vitest';

let nextId = 0;
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:${nextId++}`,
}));

const sdxlModel = { key: 'sdxl-model', hash: 'h', name: 'SDXL', base: 'sdxl', type: 'main' };
const sd1Model = { key: 'sd1-model', hash: 'h', name: 'SD1.5', base: 'sd-1', type: 'main' };
const upscaleModel = {
  key: 'spandrel-key',
  hash: 'h',
  name: 'RealESRGAN',
  base: 'any',
  type: 'spandrel_image_to_image',
};
const tileControlnetModel = { key: 'tile-cn-key', name: 'Tile CN', base: 'sdxl', type: 'controlnet' };
const upscaleInitialImage = { image_name: 'test.png', width: 512, height: 512 };

vi.mock('features/metadata/util/modelFetchingHelpers', () => ({
  fetchModelConfigWithTypeGuard: vi.fn(() => Promise.resolve(sdxlModel)),
}));

vi.mock('features/nodes/util/graph/generation/addSDXLLoRAs', () => ({
  addSDXLLoRAs: vi.fn(),
}));

vi.mock('./generation/addLoRAs', () => ({
  addLoRAs: vi.fn(),
}));

vi.mock('services/api/types', async () => {
  const actual = await vi.importActual('services/api/types');
  return {
    ...actual,
    isNonRefinerMainModelConfig: vi.fn(() => true),
    isSpandrelImageToImageModelConfig: vi.fn(() => true),
  };
});

vi.mock('./graphBuilderUtils', () => ({
  getBoardField: vi.fn(() => undefined),
  selectPresetModifiedPrompts: vi.fn(() => ({
    positive: 'a prompt',
    negative: 'a negative prompt',
  })),
}));

import { buildMultidiffusionUpscaleGraph } from './buildMultidiffusionUpscaleGraph';

const defaultParams = {
  model: sdxlModel,
  upscaleCfgScale: 7,
  upscaleScheduler: 'euler',
  steps: 20,
  vaePrecision: 'fp16',
  vae: null,
};

const defaultUpscale = {
  upscaleModel,
  upscaleInitialImage,
  structure: 0,
  creativity: 5,
  tileControlnetModel,
  scale: 2,
  tileSize: 512,
  tileOverlap: 128,
};

type NullablePartial<T> = { [K in keyof T]?: T[K] | null };

const makeState = (overrides?: {
  params?: NullablePartial<typeof defaultParams>;
  upscale?: NullablePartial<typeof defaultUpscale>;
}) =>
  ({
    params: { ...defaultParams, ...overrides?.params },
    upscale: { ...defaultUpscale, ...overrides?.upscale },
    gallery: { autoAddBoardId: 'none' },
  }) as never;

describe('buildMultidiffusionUpscaleGraph', () => {
  afterEach(() => {
    nextId = 0;
  });

  describe('SDXL graph structure', () => {
    it('includes the expected core nodes', async () => {
      const { g } = await buildMultidiffusionUpscaleGraph(makeState());
      const graph = g.getGraph();
      const types = Object.values(graph.nodes).map((n) => n.type);

      expect(types).toContain('spandrel_image_to_image_autoscale');
      expect(types).toContain('unsharp_mask');
      expect(types).toContain('sdxl_model_loader');
      expect(types).toContain('sdxl_compel_prompt');
      expect(types).toContain('i2l');
      expect(types).toContain('tiled_multi_diffusion_denoise_latents');
      expect(types).toContain('l2i');
      expect(types).toContain('controlnet');
    });

    it('sets the l2i node as non-intermediate (final output)', async () => {
      const { g } = await buildMultidiffusionUpscaleGraph(makeState());
      const graph = g.getGraph();
      const l2i = Object.values(graph.nodes).find((n) => n.type === 'l2i');

      expect(l2i?.is_intermediate).toBe(false);
    });

    it('includes two controlnet nodes for tile guidance', async () => {
      const { g } = await buildMultidiffusionUpscaleGraph(makeState());
      const graph = g.getGraph();
      const cnNodes = Object.values(graph.nodes).filter((n) => n.type === 'controlnet');

      expect(cnNodes).toHaveLength(2);
    });
  });

  describe('SD1.5 graph structure', () => {
    it('uses main_model_loader and clip_skip for SD1.5', async () => {
      const { g } = await buildMultidiffusionUpscaleGraph(makeState({ params: { model: sd1Model } }));
      const graph = g.getGraph();
      const types = Object.values(graph.nodes).map((n) => n.type);

      expect(types).toContain('main_model_loader');
      expect(types).toContain('clip_skip');
      expect(types).toContain('compel');
      expect(types).not.toContain('sdxl_model_loader');
    });
  });

  describe('assertions', () => {
    it('throws when model is missing', async () => {
      await expect(buildMultidiffusionUpscaleGraph(makeState({ params: { model: null } }))).rejects.toThrow();
    });

    it('throws when tileControlnetModel is missing', async () => {
      await expect(
        buildMultidiffusionUpscaleGraph(makeState({ upscale: { tileControlnetModel: null } }))
      ).rejects.toThrow();
    });

    it('throws when upscaleInitialImage is missing', async () => {
      await expect(
        buildMultidiffusionUpscaleGraph(makeState({ upscale: { upscaleInitialImage: null } }))
      ).rejects.toThrow();
    });
  });
});
