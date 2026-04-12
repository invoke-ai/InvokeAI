import { afterEach, describe, expect, it, vi } from 'vitest';

let nextId = 0;
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:${nextId++}`,
}));

const fluxModel = { key: 'flux-model', hash: 'h', name: 'FLUX Dev', base: 'flux', type: 'main' };
const upscaleModel = {
  key: 'spandrel-key',
  hash: 'h',
  name: 'RealESRGAN',
  base: 'any',
  type: 'spandrel_image_to_image',
};
const t5EncoderModel = { key: 't5-key', name: 'T5-XXL', base: 'any', type: 't5_encoder' };
const clipEmbedModel = { key: 'clip-key', name: 'CLIP Embed', base: 'any', type: 'clip_embed' };
const fluxVAE = { key: 'flux-vae-key', name: 'FLUX VAE', base: 'any', type: 'vae' };
const tileControlnetModel = { key: 'tile-cn-key', name: 'FLUX Tile CN', base: 'flux', type: 'controlnet' };
const upscaleInitialImage = { image_name: 'test.png', width: 512, height: 512 };

vi.mock('features/metadata/util/modelFetchingHelpers', () => ({
  fetchModelConfigWithTypeGuard: vi.fn(() => Promise.resolve(fluxModel)),
}));

vi.mock('features/nodes/util/graph/generation/addFLUXLoRAs', () => ({
  addFLUXLoRAs: vi.fn(),
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

import { buildFluxMultidiffusionUpscaleGraph } from './buildFluxMultidiffusionUpscaleGraph';

const makeState = (overrides?: { tileControlnetModel?: typeof tileControlnetModel | null }) =>
  ({
    params: {
      model: fluxModel,
      steps: 20,
      guidance: 3.5,
      fluxScheduler: 'euler',
      t5EncoderModel,
      clipEmbedModel,
      fluxVAE,
    },
    upscale: {
      upscaleModel,
      upscaleInitialImage,
      structure: 0,
      creativity: 5,
      tileControlnetModel:
        overrides?.tileControlnetModel !== undefined ? overrides.tileControlnetModel : tileControlnetModel,
      scale: 2,
      tileSize: 512,
      tileOverlap: 128,
    },
    gallery: { autoAddBoardId: 'none' },
  }) as never;

describe('buildFluxMultidiffusionUpscaleGraph', () => {
  afterEach(() => {
    nextId = 0;
  });

  describe('graph structure', () => {
    it('includes the expected core nodes', async () => {
      const { g } = await buildFluxMultidiffusionUpscaleGraph(makeState());
      const graph = g.getGraph();
      const types = Object.values(graph.nodes).map((n) => n.type);

      expect(types).toContain('spandrel_image_to_image_autoscale');
      expect(types).toContain('unsharp_mask');
      expect(types).toContain('flux_model_loader');
      expect(types).toContain('flux_text_encoder');
      expect(types).toContain('flux_vae_encode');
      expect(types).toContain('tiled_flux_denoise');
      expect(types).toContain('flux_vae_decode');
    });

    it('includes controlnet nodes when tileControlnetModel is set', async () => {
      const { g } = await buildFluxMultidiffusionUpscaleGraph(makeState());
      const graph = g.getGraph();
      const types = Object.values(graph.nodes).map((n) => n.type);

      expect(types).toContain('flux_controlnet');
    });

    it('omits controlnet nodes when tileControlnetModel is null', async () => {
      const { g } = await buildFluxMultidiffusionUpscaleGraph(makeState({ tileControlnetModel: null }));
      const graph = g.getGraph();
      const types = Object.values(graph.nodes).map((n) => n.type);

      expect(types).not.toContain('flux_controlnet');
    });

    it('connects spandrel output to unsharp mask', async () => {
      const { g } = await buildFluxMultidiffusionUpscaleGraph(makeState());
      const graph = g.getGraph();

      expect(graph.edges).toContainEqual(
        expect.objectContaining({
          source: expect.objectContaining({ field: 'image' }),
          destination: expect.objectContaining({ field: 'image' }),
        })
      );
    });

    it('sets the vae decode node as non-intermediate (final output)', async () => {
      const { g } = await buildFluxMultidiffusionUpscaleGraph(makeState());
      const graph = g.getGraph();
      const vaeDecode = Object.values(graph.nodes).find((n) => n.type === 'flux_vae_decode');

      expect(vaeDecode?.is_intermediate).toBe(false);
    });
  });

  describe('assertions', () => {
    it('throws when model is missing', async () => {
      const state = makeState();
      (state as never as Record<string, Record<string, unknown>>).params.model = null;
      await expect(buildFluxMultidiffusionUpscaleGraph(state)).rejects.toThrow();
    });

    it('throws when upscaleModel is missing', async () => {
      const state = makeState();
      (state as never as Record<string, Record<string, unknown>>).upscale.upscaleModel = null;
      await expect(buildFluxMultidiffusionUpscaleGraph(state)).rejects.toThrow();
    });

    it('throws when t5EncoderModel is missing', async () => {
      const state = makeState();
      (state as never as Record<string, Record<string, unknown>>).params.t5EncoderModel = null;
      await expect(buildFluxMultidiffusionUpscaleGraph(state)).rejects.toThrow();
    });
  });
});
