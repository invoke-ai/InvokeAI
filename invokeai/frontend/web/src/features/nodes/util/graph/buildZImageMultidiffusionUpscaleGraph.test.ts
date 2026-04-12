import { assert } from 'tsafe';
import { afterEach, describe, expect, it, vi } from 'vitest';

let nextId = 0;
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:${nextId++}`,
}));

const zImageModel = { key: 'z-image-model', hash: 'h', name: 'Z-Image', base: 'z-image', type: 'main' };
const upscaleModel = {
  key: 'spandrel-key',
  hash: 'h',
  name: 'RealESRGAN',
  base: 'any',
  type: 'spandrel_image_to_image',
};
const zImageVaeModel = { key: 'vae-key', name: 'FLUX VAE', base: 'any', type: 'vae' };
const zImageQwen3EncoderModel = { key: 'qwen3-key', name: 'Qwen3 Encoder', base: 'any', type: 'qwen3_encoder' };
const tileControlnetModel = { key: 'tile-cn-key', name: 'Z-Image Tile CN', base: 'z-image', type: 'controlnet' };
const upscaleInitialImage = { image_name: 'test.png', width: 512, height: 512 };

vi.mock('features/metadata/util/modelFetchingHelpers', () => ({
  fetchModelConfigWithTypeGuard: vi.fn(() => Promise.resolve(zImageModel)),
}));

vi.mock('features/nodes/util/graph/generation/addZImageLoRAs', () => ({
  addZImageLoRAs: vi.fn(),
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
}));

import { buildZImageMultidiffusionUpscaleGraph } from './buildZImageMultidiffusionUpscaleGraph';

const defaultParams = {
  model: zImageModel,
  steps: 20,
  zImageScheduler: 'euler',
  zImageVaeModel,
  zImageQwen3EncoderModel,
  zImageQwen3SourceModel: null,
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

describe('buildZImageMultidiffusionUpscaleGraph', () => {
  afterEach(() => {
    nextId = 0;
  });

  describe('graph structure', () => {
    it('includes the expected core nodes', async () => {
      const { g } = await buildZImageMultidiffusionUpscaleGraph(makeState());
      const graph = g.getGraph();
      const types = Object.values(graph.nodes).map((n) => n.type);

      expect(types).toContain('spandrel_image_to_image_autoscale');
      expect(types).toContain('unsharp_mask');
      expect(types).toContain('z_image_model_loader');
      expect(types).toContain('z_image_text_encoder');
      expect(types).toContain('z_image_i2l');
      expect(types).toContain('tiled_z_image_denoise');
      expect(types).toContain('z_image_l2i');
    });

    it('includes controlnet node when tileControlnetModel is set', async () => {
      const { g } = await buildZImageMultidiffusionUpscaleGraph(makeState());
      const graph = g.getGraph();
      const types = Object.values(graph.nodes).map((n) => n.type);

      expect(types).toContain('z_image_control');
    });

    it('omits controlnet node when tileControlnetModel is null', async () => {
      const { g } = await buildZImageMultidiffusionUpscaleGraph(makeState({ upscale: { tileControlnetModel: null } }));
      const graph = g.getGraph();
      const types = Object.values(graph.nodes).map((n) => n.type);

      expect(types).not.toContain('z_image_control');
    });

    it('sets guidance_scale to 1.0 for Z-Image Turbo', async () => {
      const { g } = await buildZImageMultidiffusionUpscaleGraph(makeState());
      const graph = g.getGraph();
      const denoise = Object.values(graph.nodes).find((n) => n.type === 'tiled_z_image_denoise');

      assert(denoise && 'guidance_scale' in denoise);
      expect(denoise.guidance_scale).toBe(1.0);
    });

    it('sets the l2i node as non-intermediate (final output)', async () => {
      const { g } = await buildZImageMultidiffusionUpscaleGraph(makeState());
      const graph = g.getGraph();
      const l2i = Object.values(graph.nodes).find((n) => n.type === 'z_image_l2i');

      expect(l2i?.is_intermediate).toBe(false);
    });
  });

  describe('assertions', () => {
    it('throws when model is missing', async () => {
      await expect(buildZImageMultidiffusionUpscaleGraph(makeState({ params: { model: null } }))).rejects.toThrow();
    });

    it('throws when upscaleModel is missing', async () => {
      await expect(
        buildZImageMultidiffusionUpscaleGraph(makeState({ upscale: { upscaleModel: null } }))
      ).rejects.toThrow();
    });

    it('throws when upscaleInitialImage is missing', async () => {
      await expect(
        buildZImageMultidiffusionUpscaleGraph(makeState({ upscale: { upscaleInitialImage: null } }))
      ).rejects.toThrow();
    });
  });
});
