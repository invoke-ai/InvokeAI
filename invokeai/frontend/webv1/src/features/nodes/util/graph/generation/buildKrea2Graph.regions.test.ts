import { afterEach, describe, expect, it, vi } from 'vitest';

// Unlike buildKrea2Graph.test.ts, this suite deliberately runs the REAL addRegions, addKrea2LoRAs and
// regional-guidance validators so the composed canvas path is covered end to end.

vi.mock('app/logging/logger', () => ({
  logger: () => ({
    debug: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  }),
}));

let nextId = 0;
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:${nextId++}`,
}));

const model = {
  key: 'krea2-model',
  hash: 'krea2-hash',
  name: 'Krea-2 Turbo',
  base: 'krea-2',
  type: 'main',
  format: 'diffusers',
  variant: 'krea2_turbo',
};

const defaultParams = {
  cfgScale: 1 as number | number[],
  steps: 8,
  krea2VaeModel: null as unknown,
  krea2Qwen3VlEncoderModel: null as unknown,
  krea2RebalanceEnabled: false,
  krea2RebalanceMultiplier: 4,
  krea2RebalanceWeights: '1,1,1,1,1,1,1,2.5,5,1.1,4,1',
  krea2SeedVarianceEnabled: false,
  krea2SeedVarianceStrength: 0.5,
  krea2SeedVarianceRandomizePercent: 50,
};

let params = { ...defaultParams };
let regions: unknown[] = [];

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn(() => model),
  selectParamsSlice: vi.fn(() => params),
}));

let refImageEntities: unknown[] = [];
vi.mock('features/controlLayers/store/refImagesSlice', async () => {
  const actual = await vi.importActual('features/controlLayers/store/refImagesSlice');
  return { ...actual, selectRefImagesSlice: vi.fn(() => ({ entities: refImageEntities })) };
});

vi.mock('features/controlLayers/store/validators', async () => {
  const actual = await vi.importActual('features/controlLayers/store/validators');
  return { ...actual, getGlobalReferenceImageWarnings: vi.fn(() => []) };
});

vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasMetadata: vi.fn(() => ({})),
  selectCanvasSlice: vi.fn(() => ({
    bbox: { rect: { x: 0, y: 0, width: 1024, height: 1024 } },
    regionalGuidance: { entities: regions },
  })),
}));

vi.mock('features/metadata/util/modelFetchingHelpers', () => ({
  fetchModelConfigWithTypeGuard: vi.fn(() => Promise.resolve(model)),
}));

vi.mock('features/nodes/util/graph/generation/addNSFWChecker', () => ({
  addNSFWChecker: vi.fn((_g, node) => node),
}));

vi.mock('features/nodes/util/graph/generation/addWatermarker', () => ({
  addWatermarker: vi.fn((_g, node) => node),
}));

vi.mock('features/nodes/util/graph/generation/addTextToImage', () => ({
  addTextToImage: vi.fn(({ l2i }) => l2i),
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  selectCanvasOutputFields: vi.fn(() => ({})),
  selectPresetModifiedPrompts: vi.fn(() => ({
    positive: 'a global prompt',
    negative: 'a negative prompt',
  })),
}));

vi.mock('features/ui/store/uiSelectors', () => ({
  selectActiveTab: vi.fn(() => 'canvas'),
}));

vi.mock('services/api/types', async () => {
  const actual = await vi.importActual('services/api/types');
  return {
    ...actual,
    isNonRefinerMainModelConfig: vi.fn(() => true),
  };
});

import { getRegionalGuidanceState } from 'features/controlLayers/store/util';

import { buildKrea2Graph } from './buildKrea2Graph';

type BuiltGraph = Awaited<ReturnType<typeof buildKrea2Graph>>['g'];

/** A region with a rasterized object, so the real validators do not reject it for being empty. */
const makeRegion = (id: string, overrides: Record<string, unknown>) => {
  const region = getRegionalGuidanceState(id, {});
  Object.assign(region, overrides);
  region.objects = [{ id: `${id}-object`, type: 'rect' }] as never;
  return region;
};

const manager = {
  id: 'manager',
  adapters: {
    regionMasks: {
      get: (id: string) => ({
        renderer: {
          rasterize: () => Promise.resolve({ image_name: `${id}.png` }),
        },
      }),
    },
  },
};

const loraState = { loras: [] as unknown[] };

const build = () =>
  buildKrea2Graph({
    generationMode: 'txt2img',
    manager: manager as never,
    state: {
      system: { shouldUseNSFWChecker: false, shouldUseWatermarker: false },
      loras: loraState,
    } as never,
  });

const nodesOf = (g: BuiltGraph) => Object.values(g.getGraph().nodes);
const regionalEncoders = (g: BuiltGraph) =>
  nodesOf(g).filter((node) => node.type === 'krea2_text_encoder' && node.id.startsWith('prompt_region_'));
const sourceOf = (g: BuiltGraph, nodeId: string, field: string) =>
  g.getGraph().edges.find((edge) => edge.destination.node_id === nodeId && edge.destination.field === field)?.source;

describe('buildKrea2Graph - regional guidance (composed)', () => {
  afterEach(() => {
    nextId = 0;
    refImageEntities = [];
    params = { ...defaultParams };
    regions = [];
    loraState.loras = [];
    vi.clearAllMocks();
  });

  it('adds only regions the validators accept and wires them into the positive collector', async () => {
    regions = [
      makeRegion('valid', { positivePrompt: 'a red fox' }),
      // Rejected by getRegionalGuidanceWarnings for Krea-2. If it were not filtered out, addRegions would
      // throw on its `assert(!isKrea2, ...)` guard, so this also proves the validator and the guard agree.
      makeRegion('with-negative', { positivePrompt: 'a blue fox', negativePrompt: 'blurry' }),
      makeRegion('with-auto-negative', { positivePrompt: 'a green fox', autoNegative: true }),
      makeRegion('empty', {}),
    ];

    const { g } = await build();
    const graph = g.getGraph();

    const encoders = regionalEncoders(g);
    expect(encoders).toHaveLength(1);
    expect(encoders[0]).toMatchObject({ prompt: 'a red fox' });

    // The regional mask comes from the valid region's rasterized image, not another region's.
    const maskSource = sourceOf(g, encoders[0]!.id, 'mask');
    expect(maskSource).toBeDefined();
    expect(graph.nodes[maskSource!.node_id]).toMatchObject({
      type: 'alpha_mask_to_tensor',
      image: { image_name: 'valid.png' },
    });

    // No canvas-side negative or auto-negative regional conditioning is produced for Krea-2.
    expect(nodesOf(g).some((node) => node.id.startsWith('prompt_region_negative_cond'))).toBe(false);
    expect(nodesOf(g).some((node) => node.type === 'invert_tensor_mask')).toBe(false);

    // Both the global and the regional conditioning land on the same collector, which feeds denoise.
    const posCondCollect = nodesOf(g).find((node) => node.id.startsWith('pos_cond_collect'));
    expect(posCondCollect).toBeDefined();
    const collectedFrom = graph.edges
      .filter((edge) => edge.destination.node_id === posCondCollect!.id && edge.destination.field === 'item')
      .map((edge) => edge.source.node_id);
    expect(collectedFrom).toHaveLength(2);
    expect(collectedFrom.some((id) => id.startsWith('pos_prompt:'))).toBe(true);
    expect(collectedFrom).toContain(encoders[0]!.id);
    expect(sourceOf(g, nodesOf(g).find((n) => n.type === 'krea2_denoise')!.id, 'positive_conditioning')).toEqual({
      node_id: posCondCollect!.id,
      field: 'collection',
    });
  });

  it('deletes the placeholder IP adapter collector and leaves no dangling edges', async () => {
    regions = [makeRegion('valid', { positivePrompt: 'a red fox' })];

    const { g } = await build();
    const graph = g.getGraph();

    expect(nodesOf(g).some((node) => node.id.startsWith('ip_adapter_collect'))).toBe(false);
    // Without LoRAs the only surviving collector is the positive conditioning one.
    expect(nodesOf(g).filter((node) => node.type === 'collect')).toHaveLength(1);
    for (const edge of graph.edges) {
      expect(graph.nodes[edge.source.node_id]).toBeDefined();
      expect(graph.nodes[edge.destination.node_id]).toBeDefined();
    }
  });

  it('routes regional encoders through the LoRA collection loader, not the raw model loader', async () => {
    // addKrea2LoRAs runs before addRegions and rewrites the global encoder's qwen3_vl_encoder edge; the
    // regional encoders are created afterwards by cloning that edge. If the ordering ever flips, the
    // regional prompts would silently be encoded without the LoRAs applied.
    regions = [makeRegion('valid', { positivePrompt: 'a red fox' })];
    loraState.loras = [
      {
        id: 'lora-1',
        isEnabled: true,
        weight: 0.8,
        model: { key: 'lora-key', hash: 'lora-hash', name: 'A LoRA', base: 'krea-2', type: 'lora' },
      },
    ];

    const { g } = await build();

    const loraLoader = nodesOf(g).find((node) => node.type === 'krea2_lora_collection_loader');
    expect(loraLoader).toBeDefined();

    const encoders = regionalEncoders(g);
    expect(encoders).toHaveLength(1);
    expect(sourceOf(g, encoders[0]!.id, 'qwen3_vl_encoder')).toEqual({
      node_id: loraLoader!.id,
      field: 'qwen3_vl_encoder',
    });
    // Sanity: the global encoder is rerouted too, so the regional edge is not accidentally matching a
    // still-unrouted graph.
    const globalEncoder = nodesOf(g).find((node) => node.id.startsWith('pos_prompt:'));
    expect(sourceOf(g, globalEncoder!.id, 'qwen3_vl_encoder')).toEqual({
      node_id: loraLoader!.id,
      field: 'qwen3_vl_encoder',
    });
  });

  it('applies the enabled conditioning enhancers to the regional prompt as well as the global one', async () => {
    params = {
      ...defaultParams,
      krea2RebalanceEnabled: true,
      krea2SeedVarianceEnabled: true,
      krea2SeedVarianceStrength: 0.5,
    };
    regions = [makeRegion('valid', { positivePrompt: 'a red fox' })];

    const { g } = await build();
    const graph = g.getGraph();

    // One enhancer chain per conditioning: global + one region.
    expect(nodesOf(g).filter((node) => node.type === 'krea2_conditioning_rebalance')).toHaveLength(2);
    expect(nodesOf(g).filter((node) => node.type === 'krea2_seed_variance')).toHaveLength(2);

    const encoder = regionalEncoders(g)[0]!;
    const rebalance = graph.edges.find(
      (edge) =>
        edge.source.node_id === encoder.id &&
        graph.nodes[edge.destination.node_id]?.type === 'krea2_conditioning_rebalance'
    );
    expect(rebalance).toBeDefined();

    const seedVariance = graph.edges.find(
      (edge) =>
        edge.source.node_id === rebalance!.destination.node_id &&
        graph.nodes[edge.destination.node_id]?.type === 'krea2_seed_variance'
    );
    expect(seedVariance).toBeDefined();

    // The end of the regional chain - not the raw encoder - is what gets collected.
    const posCondCollect = nodesOf(g).find((node) => node.id.startsWith('pos_cond_collect'))!;
    const collectedFrom = graph.edges
      .filter((edge) => edge.destination.node_id === posCondCollect.id && edge.destination.field === 'item')
      .map((edge) => edge.source.node_id);
    expect(collectedFrom).toContain(seedVariance!.destination.node_id);
    expect(collectedFrom).not.toContain(encoder.id);
  });
});
