import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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
  key: 'anima-model',
  hash: 'anima-hash',
  name: 'Anima Generate',
  base: 'anima',
  type: 'main',
};

const animaVaeModel = { key: 'anima-vae', name: 'Anima VAE', base: 'any', type: 'vae' };
const animaQwen3EncoderModel = { key: 'anima-qwen3', name: 'Qwen3 0.6B', base: 'any', type: 'qwen3_encoder' };

const defaultParams: {
  cfgScale: number | number[];
  steps: number;
} = {
  cfgScale: 4,
  steps: 20,
};

let params = { ...defaultParams };

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn(() => model),
  selectParamsSlice: vi.fn(() => params),
  selectAnimaVaeModel: vi.fn(() => animaVaeModel),
  selectAnimaQwen3EncoderModel: vi.fn(() => animaQwen3EncoderModel),
  selectAnimaScheduler: vi.fn(() => 'euler'),
}));

vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasMetadata: vi.fn(() => ({})),
  selectCanvasSlice: vi.fn(() => ({})),
}));

vi.mock('features/metadata/util/modelFetchingHelpers', () => ({
  fetchModelConfigWithTypeGuard: vi.fn(() => Promise.resolve(model)),
}));

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

vi.mock('features/nodes/util/graph/generation/addAnimaLoRAs', () => ({
  addAnimaLoRAs: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addRegions', () => ({
  addRegions: vi.fn(),
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

import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';

import { buildAnimaGraph } from './buildAnimaGraph';

describe('buildAnimaGraph', () => {
  afterEach(() => {
    nextId = 0;
    params = { ...defaultParams };
  });

  describe('CFG gating', () => {
    it('omits negative conditioning when cfgScale <= 1', async () => {
      params = { ...defaultParams, cfgScale: 1 };

      const { g } = await buildAnimaGraph({
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

    it('includes negative conditioning when cfgScale > 1', async () => {
      params = { ...defaultParams, cfgScale: 4 };

      const { g } = await buildAnimaGraph({
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

  describe('graph structure', () => {
    it('includes the anima model loader node', async () => {
      const { g } = await buildAnimaGraph({
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
      expect(nodeIds.some((id) => id.startsWith('anima_model_loader:'))).toBe(true);
    });

    it('includes the anima text encoder node', async () => {
      const { g } = await buildAnimaGraph({
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
      expect(nodeIds.some((id) => id.startsWith('pos_prompt:'))).toBe(true);
    });

    it('includes the anima denoise node', async () => {
      const { g } = await buildAnimaGraph({
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
      const nodeTypes = Object.values(graph.nodes).map((n) => n.type);
      expect(nodeTypes).toContain('anima_denoise');
    });
  });

  // The collect-node lifecycle in buildAnimaGraph: a single control_lllite collect node is created
  // when an inpaint adapter and/or control layers are present, wired into denoise.control_lllite when
  // it has at least one feeder, and otherwise never created. The per-adapter edges (image + inverted
  // mask) live inside addInpaint/addOutpaint, which are mocked here, so they are not asserted.
  describe('Anima LLLite collect node', () => {
    // Non-null manager triggers the inpaint branch; entities are empty so addAnimaLLLiteControl adds
    // nothing and never touches the manager.
    const manager = { id: 'test-manager' } as never;
    const inpaintCanvas = {
      controlLayers: { entities: [] },
      regionalGuidance: { entities: [] },
      bbox: { rect: { x: 0, y: 0, width: 512, height: 512 } },
    };

    const buildInpaintGraph = () =>
      buildAnimaGraph({
        generationMode: 'inpaint',
        manager,
        state: {
          system: {
            shouldUseNSFWChecker: false,
            shouldUseWatermarker: false,
          },
        },
      } as never);

    beforeEach(() => {
      vi.mocked(selectCanvasSlice).mockReturnValue(inpaintCanvas as never);
      // addInpaint is mocked; return the real l2i node it receives so it is a valid canvas output.
      // addInpaint really returns the paste-back node; for the test the l2i node it receives is a
      // sufficient (and valid) stand-in canvas output, hence the return cast.
      vi.mocked(addInpaint).mockImplementation((arg) => Promise.resolve(arg.l2i as never));
    });

    afterEach(() => {
      vi.mocked(selectCanvasSlice).mockReturnValue({} as never);
      vi.mocked(addInpaint).mockReset();
    });

    it('wires the collect node into denoise.control_lllite when an inpaint adapter is present', async () => {
      params = { ...defaultParams, animaLLLiteModel: { key: 'lllite-key' } } as never;

      const { g } = await buildInpaintGraph();

      const graph = g.getGraph();
      const collectId = Object.keys(graph.nodes).find((id) => id.startsWith('control_lllite_collect:'));
      const denoiseId = Object.keys(graph.nodes).find((id) => id.startsWith('denoise_latents:'));
      expect(collectId).toBeDefined();
      expect(denoiseId).toBeDefined();

      const hasEdgeToDenoise = graph.edges.some(
        (edge) =>
          edge.source.node_id === collectId &&
          edge.source.field === 'collection' &&
          edge.destination.node_id === denoiseId &&
          edge.destination.field === 'control_lllite'
      );
      expect(hasEdgeToDenoise).toBe(true);
    });

    it('creates no collect node when there is no adapter and no control layers', async () => {
      params = { ...defaultParams, animaLLLiteModel: null } as never;

      const { g } = await buildInpaintGraph();

      const graph = g.getGraph();
      const hasCollectNode = Object.keys(graph.nodes).some((id) => id.startsWith('control_lllite_collect:'));
      const hasControlLLLiteEdge = graph.edges.some((edge) => edge.destination.field === 'control_lllite');
      expect(hasCollectNode).toBe(false);
      expect(hasControlLLLiteEdge).toBe(false);
    });
  });
});
