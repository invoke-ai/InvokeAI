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

const baseModel = {
  key: 'krea2-model',
  hash: 'krea2-hash',
  name: 'Krea-2 Turbo',
  base: 'krea-2',
  type: 'main',
  format: 'diffusers',
  variant: 'krea2_turbo',
};

let model: Record<string, unknown> = { ...baseModel };

const defaultParams = {
  cfgScale: 1 as number | number[],
  steps: 8,
  krea2VaeModel: null as unknown,
  krea2Qwen3VlEncoderModel: null as unknown,
  krea2RebalanceEnabled: false,
  krea2RebalanceMultiplier: 4,
  krea2RebalanceWeights: '1,1,1,1,1,1,1,2.5,5,1.1,4,1',
  krea2SeedVarianceEnabled: false,
  krea2SeedVarianceStrength: 20,
  krea2SeedVarianceRandomizePercent: 50,
};

let params = { ...defaultParams };

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn(() => model),
  selectParamsSlice: vi.fn(() => params),
}));

vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasMetadata: vi.fn(() => ({})),
}));

vi.mock('features/metadata/util/modelFetchingHelpers', () => ({
  fetchModelConfigWithTypeGuard: vi.fn(() => Promise.resolve(model)),
}));

vi.mock('features/nodes/util/graph/generation/addImageToImage', () => ({
  addImageToImage: vi.fn(({ l2i }) => Promise.resolve(l2i)),
}));

vi.mock('features/nodes/util/graph/generation/addInpaint', () => ({
  addInpaint: vi.fn(({ l2i }) => Promise.resolve(l2i)),
}));

vi.mock('features/nodes/util/graph/generation/addOutpaint', () => ({
  addOutpaint: vi.fn(({ l2i }) => Promise.resolve(l2i)),
}));

vi.mock('features/nodes/util/graph/generation/addKrea2LoRAs', () => ({
  addKrea2LoRAs: vi.fn(),
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

import { addImageToImage } from './addImageToImage';
import { addInpaint } from './addInpaint';
import { addOutpaint } from './addOutpaint';
import { buildKrea2Graph } from './buildKrea2Graph';

type BuiltGraph = Awaited<ReturnType<typeof buildKrea2Graph>>['g'];

const buildTxt2Img = () =>
  buildKrea2Graph({
    generationMode: 'txt2img',
    manager: null,
    state: {
      system: { shouldUseNSFWChecker: false, shouldUseWatermarker: false },
    } as never,
  });

const buildCanvasMode = (generationMode: 'img2img' | 'inpaint' | 'outpaint') =>
  buildKrea2Graph({
    generationMode,
    manager: { id: 'manager' } as never,
    state: {
      system: { shouldUseNSFWChecker: false, shouldUseWatermarker: false },
    } as never,
  });

const nodeTypesOf = (g: BuiltGraph): string[] => Object.values(g.getGraph().nodes).map((n) => n.type);
const posConditioningEdge = (g: BuiltGraph) =>
  g.getGraph().edges.find((e) => e.destination.field === 'positive_conditioning');

describe('buildKrea2Graph', () => {
  afterEach(() => {
    nextId = 0;
    params = { ...defaultParams };
    model = { ...baseModel };
  });

  it('builds the core txt2img node chain', async () => {
    const { g } = await buildTxt2Img();
    const types = nodeTypesOf(g);
    expect(types).toContain('krea2_model_loader');
    expect(types).toContain('krea2_text_encoder');
    expect(types).toContain('krea2_denoise');
    // Krea-2 decodes with the Qwen-Image VAE node.
    expect(types).toContain('qwen_image_l2i');
  });

  it.each([
    ['img2img', addImageToImage],
    ['inpaint', addInpaint],
    ['outpaint', addOutpaint],
  ] as const)('builds the %s graph through its canvas integration', async (mode, integration) => {
    const { g } = await buildCanvasMode(mode);

    expect(integration).toHaveBeenCalledOnce();
    expect(nodeTypesOf(g)).toContain('qwen_image_i2l');
    expect((g.getMetadataNode() as unknown as Record<string, unknown>).generation_mode).toBe(`krea2_${mode}`);
  });

  describe('CFG gating (negative conditioning)', () => {
    // Krea-2 only adds a negative prompt + negative_conditioning edge when CFG is enabled (cfg_scale > 1).
    // The distilled Turbo checkpoint runs with CFG off (cfg_scale 1.0), so recording/encoding a negative
    // prompt would be wasted work.
    it('omits the negative prompt + edge when cfg_scale <= 1 (distilled Turbo default)', async () => {
      params = { ...defaultParams, cfgScale: 1 };
      const { g } = await buildTxt2Img();
      const graph = g.getGraph();
      const hasNegPromptNode = Object.keys(graph.nodes).some((id) => id.startsWith('neg_prompt:'));
      const hasNegEdge = graph.edges.some((e) => e.destination.field === 'negative_conditioning');
      expect(hasNegPromptNode).toBe(false);
      expect(hasNegEdge).toBe(false);
    });

    it('includes the negative prompt + edge when cfg_scale > 1 (Raw / CFG on)', async () => {
      params = { ...defaultParams, cfgScale: 4.5 };
      const { g } = await buildTxt2Img();
      const graph = g.getGraph();
      const hasNegPromptNode = Object.keys(graph.nodes).some((id) => id.startsWith('neg_prompt:'));
      const hasNegEdge = graph.edges.some((e) => e.destination.field === 'negative_conditioning');
      expect(hasNegPromptNode).toBe(true);
      expect(hasNegEdge).toBe(true);
    });
  });

  describe('conditioning enhancers', () => {
    it('inserts no enhancer nodes by default; positive conditioning flows straight to denoise', async () => {
      const { g } = await buildTxt2Img();
      const types = nodeTypesOf(g);
      expect(types).not.toContain('krea2_conditioning_rebalance');
      expect(types).not.toContain('krea2_seed_variance');
      // The edge into denoise.positive_conditioning comes directly from the text encoder.
      const edge = posConditioningEdge(g);
      expect(edge).toBeDefined();
      expect(edge!.source.node_id.startsWith('pos_prompt:')).toBe(true);
    });

    it('inserts the rebalance node and reroutes positive conditioning through it when enabled', async () => {
      params = { ...defaultParams, krea2RebalanceEnabled: true };
      const { g } = await buildTxt2Img();
      const types = nodeTypesOf(g);
      expect(types).toContain('krea2_conditioning_rebalance');
      expect(types).not.toContain('krea2_seed_variance');
      const edge = posConditioningEdge(g);
      expect(edge!.source.node_id.startsWith('krea2_rebalance:')).toBe(true);
    });

    it('inserts the seed-variance node when enabled with strength > 0', async () => {
      params = { ...defaultParams, krea2SeedVarianceEnabled: true, krea2SeedVarianceStrength: 20 };
      const { g } = await buildTxt2Img();
      expect(nodeTypesOf(g)).toContain('krea2_seed_variance');
      const edge = posConditioningEdge(g);
      expect(edge!.source.node_id.startsWith('krea2_seed_variance:')).toBe(true);
    });

    it('does not insert the seed-variance node when strength is 0 (a no-op)', async () => {
      params = { ...defaultParams, krea2SeedVarianceEnabled: true, krea2SeedVarianceStrength: 0 };
      const { g } = await buildTxt2Img();
      expect(nodeTypesOf(g)).not.toContain('krea2_seed_variance');
    });

    it('chains rebalance -> seed-variance -> denoise when both are enabled', async () => {
      params = {
        ...defaultParams,
        krea2RebalanceEnabled: true,
        krea2SeedVarianceEnabled: true,
        krea2SeedVarianceStrength: 20,
      };
      const { g } = await buildTxt2Img();
      const graph = g.getGraph();
      const types = nodeTypesOf(g);
      expect(types).toContain('krea2_conditioning_rebalance');
      expect(types).toContain('krea2_seed_variance');
      // rebalance -> seed_variance
      const rebalanceToSeed = graph.edges.find(
        (e) =>
          e.source.node_id.startsWith('krea2_rebalance:') && e.destination.node_id.startsWith('krea2_seed_variance:')
      );
      expect(rebalanceToSeed).toBeDefined();
      // seed_variance -> denoise.positive_conditioning
      const edge = posConditioningEdge(g);
      expect(edge!.source.node_id.startsWith('krea2_seed_variance:')).toBe(true);
    });
  });

  describe('standalone components for non-diffusers transformers', () => {
    // A single-file / GGUF transformer has no bundled VAE or encoder, so both standalone submodels are
    // required. A Diffusers pipeline bundles them, so it needs neither.
    it('throws when a single-file/GGUF transformer has no VAE selected', async () => {
      model = { ...baseModel, format: 'gguf_quantized' };
      params = { ...defaultParams, krea2VaeModel: null, krea2Qwen3VlEncoderModel: null };
      await expect(buildTxt2Img()).rejects.toThrow(/require a VAE/);
    });

    it('throws when a single-file/GGUF transformer has no Qwen3-VL encoder selected', async () => {
      model = { ...baseModel, format: 'gguf_quantized' };
      params = {
        ...defaultParams,
        krea2VaeModel: { key: 'vae', hash: 'h', name: 'VAE', base: 'qwen-image', type: 'vae' },
        krea2Qwen3VlEncoderModel: null,
      };
      await expect(buildTxt2Img()).rejects.toThrow(/require a Qwen3-VL encoder/);
    });

    it('passes the standalone submodels to the model loader when provided', async () => {
      model = { ...baseModel, format: 'gguf_quantized' };
      params = {
        ...defaultParams,
        krea2VaeModel: { key: 'vae', hash: 'h', name: 'VAE', base: 'qwen-image', type: 'vae' },
        krea2Qwen3VlEncoderModel: { key: 'enc', hash: 'h', name: 'Enc', base: 'any', type: 'qwen3_vl_encoder' },
      };
      const { g } = await buildTxt2Img();
      const loader = Object.values(g.getGraph().nodes).find((n) => n.type === 'krea2_model_loader') as
        | { vae_model?: { key: string }; qwen3_vl_encoder_model?: { key: string } }
        | undefined;
      expect(loader?.vae_model).toMatchObject({ key: 'vae' });
      expect(loader?.qwen3_vl_encoder_model).toMatchObject({ key: 'enc' });
    });
  });

  describe('metadata', () => {
    it('records the conditioning-enhancer settings and generation mode', async () => {
      params = {
        ...defaultParams,
        krea2RebalanceEnabled: true,
        krea2RebalanceMultiplier: 4,
        krea2SeedVarianceEnabled: false,
      };
      const { g } = await buildTxt2Img();
      const metadata = g.getMetadataNode() as unknown as Record<string, unknown>;
      expect(metadata.krea2_rebalance_enabled).toBe(true);
      expect(metadata.krea2_rebalance_multiplier).toBe(4);
      expect(metadata.krea2_seed_variance_enabled).toBe(false);
      expect(metadata.generation_mode).toBe('krea2_txt2img');
    });

    it('does not record a negative prompt for the CFG-off (Turbo) default', async () => {
      params = { ...defaultParams, cfgScale: 1 };
      const { g } = await buildTxt2Img();
      const metadata = g.getMetadataNode() as unknown as Record<string, unknown>;
      expect(metadata.negative_prompt).toBeUndefined();
    });
  });
});
