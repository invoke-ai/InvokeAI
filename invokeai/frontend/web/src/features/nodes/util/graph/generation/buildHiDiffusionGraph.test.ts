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

const sd1Model = {
  key: 'sd1-model',
  hash: 'sd1-hash',
  name: 'SD 1.5',
  base: 'sd-1',
  type: 'main',
};

const sdxlModel = {
  key: 'sdxl-model',
  hash: 'sdxl-hash',
  name: 'SDXL',
  base: 'sdxl',
  type: 'main',
};

const defaultParams = {
  cfgScale: 7.5,
  cfgRescaleMultiplier: 0,
  hiDiffusionEnabled: false,
  hiDiffusionRauNetEnabled: false,
  hiDiffusionT1Ratio: 0.25,
  hiDiffusionT2Ratio: 0.1,
  hiDiffusionWindowAttnEnabled: false,
  scheduler: 'euler',
  steps: 20,
  clipSkip: 0,
  shouldUseCpuNoise: false,
  vaePrecision: 'fp16',
  vae: null,
  colorCompensation: false,
  refinerModel: null,
};

let currentModel: typeof sd1Model | typeof sdxlModel = sd1Model;
let params = { ...defaultParams };

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn(() => currentModel),
  selectParamsSlice: vi.fn(() => params),
}));

vi.mock('features/controlLayers/store/refImagesSlice', () => ({
  selectRefImagesSlice: vi.fn(() => ({ entities: [] })),
}));

vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasSlice: vi.fn(() => ({
    bbox: { rect: { x: 0, y: 0, width: 1024, height: 1024 } },
    controlLayers: { entities: [] },
    regionalGuidance: { entities: [] },
  })),
  selectCanvasMetadata: vi.fn(() => ({})),
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
  selectCanvasOutputFields: vi.fn(() => ({})),
  selectPresetModifiedPrompts: vi.fn(() => ({
    positive: 'a prompt',
    negative: 'a negative prompt',
  })),
}));

vi.mock('features/ui/store/uiSelectors', () => ({
  selectActiveTab: vi.fn(() => 'generation'),
}));

vi.mock('./addRegions', () => ({
  addRegions: vi.fn(() => Promise.resolve({ addedRegions: 0 })),
}));

import type { GraphBuilderArg } from 'features/nodes/util/graph/types';

import { buildSD1Graph } from './buildSD1Graph';
import type { Graph } from './Graph';
import { buildSDXLGraph } from './buildSDXLGraph';

const buildGraphArg = (): GraphBuilderArg =>
  ({
    generationMode: 'txt2img',
    manager: null,
    state: {
      system: {
        shouldUseNSFWChecker: false,
        shouldUseWatermarker: false,
      },
    },
  }) as unknown as GraphBuilderArg;

const getMetadata = (g: Graph): Record<string, unknown> =>
  (g as unknown as { getMetadataNode: () => Record<string, unknown> }).getMetadataNode();

const resetState = () => {
  nextId = 0;
  currentModel = sd1Model;
  params = { ...defaultParams };
};

beforeEach(resetState);
afterEach(resetState);

describe('HiDiffusion graph metadata', () => {
  it('persists disabled HiDiffusion settings in the SD1 metadata node', async () => {
    currentModel = sd1Model;

    const { g } = await buildSD1Graph(buildGraphArg());
    const metadata = getMetadata(g);

    expect(metadata.hidiffusion).toBe(false);
    expect(metadata.hidiffusion_raunet).toBe(false);
    expect(metadata.hidiffusion_window_attn).toBe(false);
    expect(metadata.hidiffusion_t1_ratio).toBe(0.25);
    expect(metadata.hidiffusion_t2_ratio).toBe(0.1);
  });

  it('persists disabled HiDiffusion settings in the SDXL metadata node', async () => {
    currentModel = sdxlModel;

    const { g } = await buildSDXLGraph(buildGraphArg());
    const metadata = getMetadata(g);

    expect(metadata.hidiffusion).toBe(false);
    expect(metadata.hidiffusion_raunet).toBe(false);
    expect(metadata.hidiffusion_window_attn).toBe(false);
    expect(metadata.hidiffusion_t1_ratio).toBe(0.25);
    expect(metadata.hidiffusion_t2_ratio).toBe(0.1);
  });
});
