import type { RootState } from 'app/store/store';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('app/logging/logger', () => ({
  logger: () => ({
    debug: vi.fn(),
  }),
}));

let nextId = 0;
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:${nextId++}`,
}));

const models = {
  sd1: {
    key: 'sd1-model',
    hash: 'sd1-hash',
    name: 'SD 1.5',
    base: 'sd-1',
    type: 'main',
  },
  sd3: {
    key: 'sd3-model',
    hash: 'sd3-hash',
    name: 'SD3',
    base: 'sd-3',
    type: 'main',
  },
  sdxl: {
    key: 'sdxl-model',
    hash: 'sdxl-hash',
    name: 'SDXL',
    base: 'sdxl',
    type: 'main',
  },
} as const;

const upscaleModel = {
  key: 'upscale-model',
  hash: 'upscale-hash',
  name: 'Upscale',
  base: 'any',
  type: 'spandrel_image_to_image',
} as const;

const controlnetModel = {
  key: 'controlnet-model',
  hash: 'controlnet-hash',
  name: 'Tile ControlNet',
  base: 'sd-1',
  type: 'controlnet',
} as const;

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn((state: RootState) => state.params.model),
  selectParamsSlice: vi.fn((state: RootState) => state.params),
}));

vi.mock('features/controlLayers/store/refImagesSlice', () => ({
  selectRefImagesSlice: vi.fn(() => ({ entities: [] })),
}));

vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasMetadata: vi.fn(() => ({})),
  selectCanvasSlice: vi.fn(() => ({
    bbox: { rect: { x: 0, y: 0, width: 1024, height: 1024 } },
    controlLayers: { entities: [] },
    regionalGuidance: { entities: [] },
  })),
}));

vi.mock('features/metadata/util/modelFetchingHelpers', () => ({
  fetchModelConfigWithTypeGuard: vi.fn((key: string) => {
    if (key === upscaleModel.key) {
      return Promise.resolve(upscaleModel);
    }
    return Promise.resolve(models.sdxl);
  }),
}));

vi.mock('features/nodes/util/graph/generation/addControlAdapters', () => ({
  addControlNets: vi.fn(() => Promise.resolve({ addedControlNets: 0 })),
  addT2IAdapters: vi.fn(() => Promise.resolve({ addedT2IAdapters: 0 })),
}));

vi.mock('features/nodes/util/graph/generation/addImageToImage', () => ({
  addImageToImage: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addInpaint', () => ({
  addInpaint: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addIPAdapters', () => ({
  addIPAdapters: vi.fn(() => ({ addedIPAdapters: 0 })),
}));

vi.mock('features/nodes/util/graph/generation/addLoRAs', () => ({
  addLoRAs: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addNSFWChecker', () => ({
  addNSFWChecker: vi.fn((_g, node) => node),
}));

vi.mock('features/nodes/util/graph/generation/addOutpaint', () => ({
  addOutpaint: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addRegions', () => ({
  addRegions: vi.fn(() => []),
}));

vi.mock('features/nodes/util/graph/generation/addSDXLLoRAs', () => ({
  addSDXLLoRAs: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addSDXLRefiner', () => ({
  addSDXLRefiner: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addSeamless', () => ({
  addSeamless: vi.fn(() => null),
}));

vi.mock('features/nodes/util/graph/generation/addTextToImage', () => ({
  addTextToImage: vi.fn(({ l2i }) => l2i),
}));

vi.mock('features/nodes/util/graph/generation/addWatermarker', () => ({
  addWatermarker: vi.fn((_g, node) => node),
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  getBoardField: vi.fn(() => undefined),
  selectCanvasOutputFields: vi.fn(() => ({})),
  selectPresetModifiedPrompts: vi.fn(() => ({
    positive: 'preset positive prompt',
    negative: 'preset negative prompt',
  })),
}));

vi.mock('features/ui/store/uiSelectors', () => ({
  selectActiveTab: vi.fn(() => 'generate'),
}));

import type { BaseModelType } from 'features/nodes/types/common';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildMultidiffusionUpscaleGraph } from 'features/nodes/util/graph/buildMultidiffusionUpscaleGraph';
import type { GraphBuilderReturn } from 'features/nodes/util/graph/types';

import { buildSD1Graph } from './buildSD1Graph';
import { buildSD3Graph } from './buildSD3Graph';
import { buildSDXLGraph } from './buildSDXLGraph';
import type { GraphType } from './Graph';

type TestNode = { id: string; type: string; [key: string]: unknown };

const buildState = (model: (typeof models)[keyof typeof models]): RootState =>
  ({
    dynamicPrompts: {
      prompts: ['positive prompt 1', 'positive prompt 2'],
      seedBehaviour: 'PER_PROMPT',
    },
    gallery: {
      autoAddBoardId: 'none',
    },
    params: {
      cfgRescaleMultiplier: 0,
      cfgScale: 7,
      clipGEmbedModel: null,
      clipLEmbedModel: null,
      clipSkip: 0,
      colorCompensation: false,
      iterations: 2,
      model,
      negativePrompt: 'raw negative prompt',
      positivePrompt: 'raw positive prompt',
      refinerModel: null,
      scheduler: 'euler',
      seed: 123,
      shouldRandomizeSeed: false,
      shouldUseCpuNoise: true,
      steps: 20,
      t5EncoderModel: null,
      upscaleCfgScale: 2,
      upscaleScheduler: 'euler',
      vae: null,
      vaePrecision: 'fp32',
    },
    system: {
      shouldUseNSFWChecker: false,
      shouldUseWatermarker: false,
    },
    upscale: {
      creativity: 4,
      scale: 2,
      structure: 5,
      tileControlnetModel: controlnetModel,
      tileOverlap: 64,
      tileSize: 512,
      upscaleInitialImage: {
        height: 512,
        image_name: 'initial.png',
        width: 512,
      },
      upscaleModel,
    },
  }) as unknown as RootState;

const getNodeByPrefix = (graph: GraphType, prefix: string) => {
  const nodes = graph.nodes as Record<string, TestNode>;
  return Object.entries(nodes).find(([id]) => id.startsWith(prefix))?.[1];
};

const expectNegativePromptWiring = (graph: GraphType, negativePromptId: string, conditioningFields: string[]) => {
  const negCond = getNodeByPrefix(graph, 'neg_cond:');
  expect(negCond).toBeDefined();
  if (!negCond) {
    throw new Error('Expected negative conditioning node to exist');
  }
  expect(negCond).not.toHaveProperty('prompt');
  expect(negCond).not.toHaveProperty('style');

  for (const field of conditioningFields) {
    expect(graph.edges).toContainEqual({
      destination: { field, node_id: negCond?.id },
      source: { field: 'value', node_id: negativePromptId },
    });
  }

  const nodes = graph.nodes as Record<string, TestNode>;
  const metadataNode = Object.values(nodes).find((node) => node.type === 'core_metadata');
  expect(metadataNode).toBeDefined();
  if (!metadataNode) {
    throw new Error('Expected metadata node to exist');
  }
  expect(metadataNode).not.toHaveProperty('negative_prompt');
  expect(graph.edges).toContainEqual({
    destination: { field: 'negative_prompt', node_id: metadataNode?.id },
    source: { field: 'value', node_id: negativePromptId },
  });
};

const expectNegativePromptBatching = (state: RootState, graphBuilderReturn: GraphBuilderReturn) => {
  expect(graphBuilderReturn.negativePrompt).toBeDefined();

  const batchConfig = prepareLinearUIBatch({
    state,
    g: graphBuilderReturn.g,
    base: (state.params.model?.base ?? 'sdxl') as BaseModelType,
    prepend: false,
    seedNode: graphBuilderReturn.seed,
    positivePromptNode: graphBuilderReturn.positivePrompt,
    negativePromptNode: graphBuilderReturn.negativePrompt,
    origin: 'test',
    destination: 'test',
  });

  const negativePromptBatchDatum = batchConfig.batch.data
    ?.flat()
    .find((datum) => datum.node_path === graphBuilderReturn.negativePrompt?.id);

  expect(negativePromptBatchDatum).toEqual({
    field_name: 'value',
    items: ['preset negative prompt', 'preset negative prompt', 'preset negative prompt', 'preset negative prompt'],
    node_path: graphBuilderReturn.negativePrompt?.id,
  });
};

beforeEach(() => {
  nextId = 0;
});

describe('SD negative prompt graph wiring', () => {
  it('wires SD1 negative prompt through a string node into conditioning, metadata, and batch data', async () => {
    const state = buildState(models.sd1);
    const result = await buildSD1Graph({ generationMode: 'txt2img', manager: null, state });
    const graph = result.g.getGraph();

    expect(result.negativePrompt).toBeDefined();
    expectNegativePromptWiring(graph, result.negativePrompt?.id ?? '', ['prompt']);
    expectNegativePromptBatching(state, result);
  });

  it('wires SDXL negative prompt through a string node into conditioning, metadata, and batch data', async () => {
    const state = buildState(models.sdxl);
    const result = await buildSDXLGraph({ generationMode: 'txt2img', manager: null, state });
    const graph = result.g.getGraph();

    expect(result.negativePrompt).toBeDefined();
    expectNegativePromptWiring(graph, result.negativePrompt?.id ?? '', ['prompt', 'style']);
    expectNegativePromptBatching(state, result);
  });

  it('wires SD3 negative prompt through a string node into conditioning, metadata, and batch data', async () => {
    const state = buildState(models.sd3);
    const result = await buildSD3Graph({ generationMode: 'txt2img', manager: null, state });
    const graph = result.g.getGraph();

    expect(result.negativePrompt).toBeDefined();
    expectNegativePromptWiring(graph, result.negativePrompt?.id ?? '', ['prompt']);
    expectNegativePromptBatching(state, result);
  });

  it.each([
    { conditioningFields: ['prompt'], model: models.sd1 },
    { conditioningFields: ['prompt', 'style'], model: models.sdxl },
  ])(
    'wires $model.base multidiffusion upscale negative prompt through a string node into conditioning, metadata, and batch data',
    async ({ conditioningFields, model }) => {
      const state = buildState(model);
      const result = await buildMultidiffusionUpscaleGraph(state);
      const graph = result.g.getGraph();

      expect(result.negativePrompt).toBeDefined();
      expectNegativePromptWiring(graph, result.negativePrompt?.id ?? '', conditioningFields);
      expectNegativePromptBatching(state, result);
    }
  );
});
