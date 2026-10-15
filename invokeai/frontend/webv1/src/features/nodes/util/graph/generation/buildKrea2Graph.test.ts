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
  krea2SeedVarianceStrength: 0.5,
  krea2SeedVarianceRandomizePercent: 50,
};

let params = { ...defaultParams };

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
    regionalGuidance: { entities: [] },
  })),
}));

vi.mock('features/metadata/util/modelFetchingHelpers', () => ({
  fetchModelConfigWithTypeGuard: vi.fn(() => Promise.resolve(model)),
}));

// The real mode helpers are what assign the denoise dimensions - anything the builder reads off `denoise`
// before they run is `undefined`. The mocks mirror that so ordering bugs are visible here (review 4977240290).
let scaledSize = { width: 1024, height: 1024 };
const applyScaledSize = (denoise: { width?: number; height?: number }) => {
  denoise.width = scaledSize.width;
  denoise.height = scaledSize.height;
};

vi.mock('features/nodes/util/graph/generation/addImageToImage', () => ({
  addImageToImage: vi.fn(({ denoise, l2i }) => {
    applyScaledSize(denoise);
    return Promise.resolve(l2i);
  }),
}));

vi.mock('features/nodes/util/graph/generation/addInpaint', () => ({
  addInpaint: vi.fn(({ denoise, l2i }) => {
    applyScaledSize(denoise);
    return Promise.resolve(l2i);
  }),
}));

vi.mock('features/nodes/util/graph/generation/addOutpaint', () => ({
  addOutpaint: vi.fn(({ denoise, l2i }) => {
    applyScaledSize(denoise);
    return Promise.resolve(l2i);
  }),
}));

vi.mock('features/nodes/util/graph/generation/addRegions', () => ({
  addRegions: vi.fn(() => Promise.resolve([])),
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
  addTextToImage: vi.fn(({ denoise, l2i }) => {
    applyScaledSize(denoise);
    return l2i;
  }),
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
import { addRegions } from './addRegions';
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
    vi.clearAllMocks();
    nextId = 0;
    refImageEntities = [];
    scaledSize = { width: 1024, height: 1024 };
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
    expect(addRegions).toHaveBeenCalledOnce();
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
    it('inserts no enhancer nodes by default; positive conditioning flows through the regional collector', async () => {
      const { g } = await buildTxt2Img();
      const types = nodeTypesOf(g);
      expect(types).not.toContain('krea2_conditioning_rebalance');
      expect(types).not.toContain('krea2_seed_variance');
      const edge = posConditioningEdge(g);
      expect(edge).toBeDefined();
      expect(edge!.source.node_id.startsWith('pos_cond_collect:')).toBe(true);
      expect(
        g
          .getGraph()
          .edges.some(
            (candidate) =>
              candidate.source.node_id.startsWith('pos_prompt:') &&
              candidate.destination.node_id.startsWith('pos_cond_collect:')
          )
      ).toBe(true);
    });

    it('inserts the rebalance node and reroutes positive conditioning through it when enabled', async () => {
      params = { ...defaultParams, krea2RebalanceEnabled: true };
      const { g } = await buildTxt2Img();
      const types = nodeTypesOf(g);
      expect(types).toContain('krea2_conditioning_rebalance');
      expect(types).not.toContain('krea2_seed_variance');
      const edge = posConditioningEdge(g);
      expect(edge!.source.node_id.startsWith('pos_cond_collect:')).toBe(true);
      expect(
        g
          .getGraph()
          .edges.some(
            (candidate) =>
              candidate.source.node_id.startsWith('krea2_rebalance:') &&
              candidate.destination.node_id.startsWith('pos_cond_collect:')
          )
      ).toBe(true);
    });

    it('inserts the seed-variance node when enabled with strength > 0', async () => {
      params = { ...defaultParams, krea2SeedVarianceEnabled: true, krea2SeedVarianceStrength: 0.5 };
      const { g } = await buildTxt2Img();
      expect(nodeTypesOf(g)).toContain('krea2_seed_variance');
      const edge = posConditioningEdge(g);
      expect(edge!.source.node_id.startsWith('pos_cond_collect:')).toBe(true);
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
        krea2SeedVarianceStrength: 0.5,
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
      // seed_variance -> collector -> denoise.positive_conditioning
      const edge = posConditioningEdge(g);
      expect(edge!.source.node_id.startsWith('pos_cond_collect:')).toBe(true);
      expect(
        graph.edges.some(
          (candidate) =>
            candidate.source.node_id.startsWith('krea2_seed_variance:') &&
            candidate.destination.node_id.startsWith('pos_cond_collect:')
        )
      ).toBe(true);
    });

    it('applies enabled enhancers to regional conditioning before collection', async () => {
      params = {
        ...defaultParams,
        krea2RebalanceEnabled: true,
        krea2SeedVarianceEnabled: true,
        krea2SeedVarianceStrength: 0.5,
      };
      vi.mocked(addRegions).mockImplementationOnce((arg) => {
        const regionalPosCond = arg.g.addNode({
          type: 'krea2_text_encoder',
          id: 'regional-positive',
          prompt: 'regional prompt',
        });
        const transformRegionalPositiveConditioning = (
          arg as typeof arg & {
            transformRegionalPositiveConditioning?: (conditioning: typeof regionalPosCond) => {
              id: string;
              type: string;
            };
          }
        ).transformRegionalPositiveConditioning;
        const conditioningSource = transformRegionalPositiveConditioning?.(regionalPosCond) ?? regionalPosCond;
        arg.g.addEdgeFromObj({
          source: { node_id: conditioningSource.id, field: 'conditioning' },
          destination: { node_id: arg.posCondCollect.id, field: 'item' },
        });
        return Promise.resolve([]);
      });

      const { g } = await buildCanvasMode('img2img');
      const graph = g.getGraph();
      const regionalRebalanceEdge = graph.edges.find(
        (edge) =>
          edge.source.node_id === 'regional-positive' &&
          graph.nodes[edge.destination.node_id]?.type === 'krea2_conditioning_rebalance'
      );
      expect(regionalRebalanceEdge).toBeDefined();
      const regionalSeedVarianceEdge = graph.edges.find(
        (edge) =>
          edge.source.node_id === regionalRebalanceEdge!.destination.node_id &&
          graph.nodes[edge.destination.node_id]?.type === 'krea2_seed_variance'
      );
      expect(regionalSeedVarianceEdge).toBeDefined();
      expect(graph.edges).toContainEqual(
        expect.objectContaining({
          source: { node_id: regionalSeedVarianceEdge!.destination.node_id, field: 'conditioning' },
          destination: expect.objectContaining({ field: 'item' }),
        })
      );
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
  describe('style reference', () => {
    const styleRefEntity = (overrides: Record<string, unknown> = {}) => ({
      id: 'ref-1',
      isEnabled: true,
      config: {
        type: 'krea2_reference_image',
        styleStrength: 1,
        image: { original: { image: { image_name: 'style.png' }, width: 512, height: 512 } },
        ...overrides,
      },
    });

    const styleReferenceNode = (g: BuiltGraph) =>
      Object.values(g.getGraph().nodes).find((n) => n.type === 'krea2_style_reference');

    it('adds no style reference node when there is no reference image', async () => {
      const { g } = await buildTxt2Img();
      expect(styleReferenceNode(g)).toBeUndefined();
    });

    it('wires the style reference between the VAE and denoise', async () => {
      refImageEntities = [styleRefEntity()];
      const { g } = await buildTxt2Img();

      const styleReference = styleReferenceNode(g) as unknown as Record<string, unknown>;
      expect(styleReference).toBeDefined();
      expect(styleReference.image).toEqual({ image_name: 'style.png' });

      const edges = g.getGraph().edges;
      expect(edges.some((e) => e.destination.node_id === styleReference.id && e.destination.field === 'vae')).toBe(
        true
      );
      expect(
        edges.some((e) => e.source.node_id === styleReference.id && e.destination.field === 'style_reference')
      ).toBe(true);
    });

    // The reference's image tokens are appended to the target's, so a size mismatch is a hard error in the
    // denoise node. The generation-mode helper is what assigns the denoise dimensions, so the style node has
    // to be built after it - reading them earlier yields `undefined`, which JSON drops, silently leaving the
    // node on its 1024x1024 backend default (review 4977240290).
    it.each([
      ['1024x1024', { width: 1024, height: 1024 }],
      ['768x1024', { width: 768, height: 1024 }],
      ['1152x896', { width: 1152, height: 896 }],
    ])('encodes the reference at the denoise resolution (%s, txt2img)', async (_label, size) => {
      scaledSize = size;
      refImageEntities = [styleRefEntity()];
      const { g } = await buildTxt2Img();

      const styleReference = styleReferenceNode(g) as unknown as Record<string, unknown>;
      const denoise = Object.values(g.getGraph().nodes).find((n) => n.type === 'krea2_denoise') as unknown as Record<
        string,
        unknown
      >;
      expect(denoise.width).toBe(size.width);
      expect(styleReference.width).toBe(size.width);
      expect(styleReference.height).toBe(size.height);
      expect(styleReference.width).toBe(denoise.width);
      expect(styleReference.height).toBe(denoise.height);
    });

    it.each(['img2img', 'inpaint', 'outpaint'] as const)(
      'encodes the reference at the denoise resolution (%s)',
      async (generationMode) => {
        scaledSize = { width: 768, height: 1024 };
        refImageEntities = [styleRefEntity()];
        const { g } = await buildCanvasMode(generationMode);

        const styleReference = styleReferenceNode(g) as unknown as Record<string, unknown>;
        expect(styleReference.width).toBe(768);
        expect(styleReference.height).toBe(1024);
      }
    );

    // 0 is documented as a full bypass. Emitting the node anyway costs a VAE encode, a capture pass per step
    // and a retained K/V cache for no visible effect.
    it('adds no style reference node when the strength is 0', async () => {
      refImageEntities = [styleRefEntity({ styleStrength: 0 })];
      const { g } = await buildTxt2Img();

      expect(styleReferenceNode(g)).toBeUndefined();
      expect((g.getMetadataNode() as unknown as Record<string, unknown>).krea2_style_strength).toBeUndefined();
    });

    it('falls through to the next reference when the first has a strength of 0', async () => {
      refImageEntities = [
        styleRefEntity({ styleStrength: 0 }),
        { ...styleRefEntity({ styleStrength: 0.5 }), id: 'ref-2' },
      ];
      const { g } = await buildTxt2Img();

      const styleReference = styleReferenceNode(g) as unknown as Record<string, unknown>;
      expect(styleReference.style_strength).toBe(0.5);
    });

    it('prefers the cropped image when one exists', async () => {
      refImageEntities = [
        styleRefEntity({
          image: {
            original: { image: { image_name: 'style.png' }, width: 512, height: 512 },
            crop: { image: { image_name: 'style-cropped.png' }, width: 256, height: 256 },
          },
        }),
      ];
      const { g } = await buildTxt2Img();

      const styleReference = styleReferenceNode(g) as unknown as Record<string, unknown>;
      expect(styleReference.image).toEqual({ image_name: 'style-cropped.png' });
    });

    it('carries the style strength onto the node and into metadata', async () => {
      refImageEntities = [styleRefEntity({ styleStrength: 0.6 })];
      const { g } = await buildTxt2Img();

      const styleReference = styleReferenceNode(g) as unknown as Record<string, unknown>;
      expect(styleReference.style_strength).toBe(0.6);
      expect((g.getMetadataNode() as unknown as Record<string, unknown>).krea2_style_strength).toBe(0.6);
    });

    it('ignores disabled reference images and ones without an image', async () => {
      refImageEntities = [{ ...styleRefEntity(), isEnabled: false }, styleRefEntity({ image: null })];
      const { g } = await buildTxt2Img();
      expect(styleReferenceNode(g)).toBeUndefined();
    });

    it('uses only the first valid reference image', async () => {
      // The technique supports exactly one reference; the panel allows several.
      refImageEntities = [
        styleRefEntity({ styleStrength: 0.25 }),
        { ...styleRefEntity({ styleStrength: 0.75 }), id: 'ref-2' },
      ];
      const { g } = await buildTxt2Img();

      const styleReferenceNodes = Object.values(g.getGraph().nodes).filter((n) => n.type === 'krea2_style_reference');
      expect(styleReferenceNodes).toHaveLength(1);
      expect((styleReferenceNodes[0] as unknown as Record<string, unknown>).style_strength).toBe(0.25);
    });

    it('ignores reference images belonging to another base', async () => {
      refImageEntities = [{ id: 'ref-1', isEnabled: true, config: { type: 'wan_reference_image', image: {} } }];
      const { g } = await buildTxt2Img();
      expect(styleReferenceNode(g)).toBeUndefined();
    });
  });
});
