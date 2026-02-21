import type { RootState } from 'app/store/store';
import type { ParamsState, RefImagesState } from 'features/controlLayers/store/types';
import { imageDTOToCroppableImage, initialIPAdapter } from 'features/controlLayers/store/util';
import type {
  ExternalApiModelConfig,
  ExternalApiModelDefaultSettings,
  ExternalImageSize,
  ExternalModelCapabilities,
  ImageDTO,
  Invocation,
} from 'services/api/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { buildExternalGraph } from './buildExternalGraph';

const createExternalModel = (overrides: Partial<ExternalApiModelConfig> = {}): ExternalApiModelConfig => {
  const maxImageSize: ExternalImageSize = { width: 1024, height: 1024 };
  const defaultSettings: ExternalApiModelDefaultSettings = { width: 1024, height: 1024, steps: 30 };
  const capabilities: ExternalModelCapabilities = {
    modes: ['txt2img'],
    supports_negative_prompt: true,
    supports_reference_images: true,
    supports_seed: true,
    supports_guidance: true,
    max_image_size: maxImageSize,
  };

  return {
    key: 'external-test',
    hash: 'external:openai:gpt-image-1',
    path: 'external://openai/gpt-image-1',
    file_size: 0,
    name: 'External Test',
    description: null,
    source: 'external://openai/gpt-image-1',
    source_type: 'url',
    source_api_response: null,
    cover_image: null,
    base: 'external',
    type: 'external_image_generator',
    format: 'external_api',
    provider_id: 'openai',
    provider_model_id: 'gpt-image-1',
    capabilities,
    default_settings: defaultSettings,
    tags: ['external'],
    is_default: false,
    ...overrides,
  };
};

let mockModelConfig: ExternalApiModelConfig | null = null;
let mockParams: ParamsState;
let mockRefImages: RefImagesState;
let mockPrompts: { positive: string; negative: string };
let mockSizes: { scaledSize: { width: number; height: number } };

const mockOutputFields = {
  id: 'external_output',
  use_cache: false,
  is_intermediate: false,
  board: undefined,
};

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectModelConfig: () => mockModelConfig,
  selectParamsSlice: () => mockParams,
}));

vi.mock('features/controlLayers/store/refImagesSlice', () => ({
  selectRefImagesSlice: () => mockRefImages,
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  getOriginalAndScaledSizesForTextToImage: () => mockSizes,
  getOriginalAndScaledSizesForOtherModes: () => ({
    scaledSize: { width: 512, height: 512 },
    rect: { x: 0, y: 0, width: 512, height: 512 },
  }),
  selectCanvasOutputFields: () => mockOutputFields,
  selectPresetModifiedPrompts: () => mockPrompts,
}));

beforeEach(() => {
  mockParams = {
    steps: 20,
    guidance: 4.5,
  } as ParamsState;
  mockPrompts = { positive: 'a test prompt', negative: 'bad prompt' };
  mockSizes = { scaledSize: { width: 768, height: 512 } };

  const imageDTO = { image_name: 'ref.png', width: 64, height: 64 } as ImageDTO;
  mockRefImages = {
    selectedEntityId: null,
    isPanelOpen: false,
    entities: [
      {
        id: 'ref-image-1',
        isEnabled: true,
        config: {
          ...initialIPAdapter,
          weight: 0.5,
          image: imageDTOToCroppableImage(imageDTO),
        },
      },
    ],
  };
});

describe('buildExternalGraph', () => {
  it('builds txt2img graph with reference images and seed', async () => {
    const modelConfig = createExternalModel();
    mockModelConfig = modelConfig;

    const { g } = await buildExternalGraph({
      generationMode: 'txt2img',
      state: {} as RootState,
      manager: null,
    });
    const graph = g.getGraph();
    const externalNode = Object.values(graph.nodes).find(
      (node) => node.type === 'external_image_generation'
    ) as Invocation<'external_image_generation'>;

    expect(externalNode).toBeDefined();
    expect(externalNode.mode).toBe('txt2img');
    expect(externalNode.width).toBe(768);
    expect(externalNode.height).toBe(512);
    expect(externalNode.negative_prompt).toBe('bad prompt');
    expect(externalNode.guidance).toBe(4.5);
    expect(externalNode.reference_images?.[0]).toEqual({ image_name: 'ref.png' });
    expect(externalNode.reference_image_weights).toEqual([0.5]);

    const seedEdge = graph.edges.find((edge) => edge.destination.field === 'seed');
    expect(seedEdge).toBeDefined();
  });

  it('throws when mode is unsupported', async () => {
    const modelConfig = createExternalModel({
      capabilities: {
        modes: ['img2img'],
      },
    });
    mockModelConfig = modelConfig;

    await expect(
      buildExternalGraph({
        generationMode: 'txt2img',
        state: {} as RootState,
        manager: null,
      })
    ).rejects.toThrow('does not support txt2img');
  });
});
