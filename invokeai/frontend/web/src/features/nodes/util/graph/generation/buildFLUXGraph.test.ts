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

// --- Flux2 Klein model fixtures ---

const flux2DiffusersModel = {
  key: 'flux2-klein-diffusers',
  hash: 'flux2-diff-hash',
  name: 'FLUX.2 Klein 4B',
  base: 'flux2',
  type: 'main',
  format: 'diffusers',
  variant: 'klein_4b',
};

const flux2GGUFModel = {
  key: 'flux2-klein-gguf',
  hash: 'flux2-gguf-hash',
  name: 'FLUX.2 Klein 4B GGUF',
  base: 'flux2',
  type: 'main',
  format: 'gguf_quantized',
  variant: 'klein_4b',
};

const kleinVaeModelFixture = { key: 'klein-vae', name: 'Klein VAE', base: 'flux2', type: 'vae' };
const kleinQwen3EncoderModelFixture = {
  key: 'klein-qwen3',
  name: 'Qwen3 4B',
  base: 'flux2',
  type: 'qwen3_encoder',
};

const flux2GGUF9BModel = {
  key: 'flux2-klein-gguf-9b',
  hash: 'flux2-gguf-9b-hash',
  name: 'FLUX.2 Klein 9B GGUF',
  base: 'flux2',
  type: 'main',
  format: 'gguf_quantized',
  variant: 'klein_9b',
};

const diffusersSourceModelFixture = {
  key: 'flux2-source-diffusers',
  hash: 'flux2-src-hash',
  name: 'FLUX.2 Klein 4B Source',
  base: 'flux2',
  type: 'main',
  format: 'diffusers',
  variant: 'klein_4b',
};

const diffusers9BSourceModelFixture = {
  key: 'flux2-source-diffusers-9b',
  hash: 'flux2-src-9b-hash',
  name: 'FLUX.2 Klein 9B Source',
  base: 'flux2',
  type: 'main',
  format: 'diffusers',
  variant: 'klein_9b',
};

// --- Mutable state ---

let model: Record<string, unknown> = { ...flux2DiffusersModel };
let kleinVaeModel: Record<string, unknown> | null = null;
let kleinQwen3EncoderModel: Record<string, unknown> | null = null;
let diffusersModels: Record<string, unknown>[] = [];

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn(() => model),
  selectParamsSlice: vi.fn(() => ({
    guidance: 4,
    steps: 20,
    fluxScheduler: 'euler',
    fluxDypePreset: 'off',
    fluxDypeScale: 2.0,
    fluxDypeExponent: 2.0,
    fluxVAE: null,
    t5EncoderModel: null,
    clipEmbedModel: null,
  })),
  selectKleinVaeModel: vi.fn(() => kleinVaeModel),
  selectKleinQwen3EncoderModel: vi.fn(() => kleinQwen3EncoderModel),
}));

vi.mock('features/controlLayers/store/refImagesSlice', () => ({
  selectRefImagesSlice: vi.fn(() => ({
    entities: [],
  })),
}));

vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasMetadata: vi.fn(() => ({})),
  selectCanvasSlice: vi.fn(() => ({})),
}));

vi.mock('features/controlLayers/store/types', () => ({
  isFlux2ReferenceImageConfig: vi.fn(() => false),
  isFluxKontextReferenceImageConfig: vi.fn(() => false),
}));

vi.mock('features/controlLayers/store/validators', () => ({
  getGlobalReferenceImageWarnings: vi.fn(() => []),
}));

vi.mock('features/nodes/util/graph/generation/addFlux2KleinLoRAs', () => ({
  addFlux2KleinLoRAs: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addFLUXFill', () => ({
  addFLUXFill: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addFLUXLoRAs', () => ({
  addFLUXLoRAs: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addFLUXRedux', () => ({
  addFLUXReduxes: vi.fn(),
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

vi.mock('features/nodes/util/graph/generation/addRegions', () => ({
  addRegions: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addTextToImage', () => ({
  addTextToImage: vi.fn(({ l2i }) => l2i),
}));

vi.mock('features/nodes/util/graph/generation/addWatermarker', () => ({
  addWatermarker: vi.fn((_g, node) => node),
}));

vi.mock('features/nodes/util/graph/generation/addControlAdapters', () => ({
  addControlLoRA: vi.fn(),
  addControlNets: vi.fn(),
}));

vi.mock('features/nodes/util/graph/generation/addIPAdapters', () => ({
  addIPAdapters: vi.fn(),
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  selectCanvasOutputFields: vi.fn(() => ({})),
  selectPresetModifiedPrompts: vi.fn(() => ({
    positive: 'a prompt',
    negative: '',
  })),
}));

vi.mock('features/ui/store/uiSelectors', () => ({
  selectActiveTab: vi.fn(() => 'generation'),
}));

vi.mock('services/api/hooks/modelsByType', () => ({
  selectFlux2DiffusersModels: vi.fn(() => diffusersModels),
}));

vi.mock('services/api/types', async () => {
  const actual = await vi.importActual('services/api/types');
  return {
    ...actual,
    isNonRefinerMainModelConfig: vi.fn(() => true),
  };
});

import { buildFLUXGraph } from './buildFLUXGraph';

const buildGraphArg = () => ({
  generationMode: 'txt2img' as const,
  manager: null,
  state: {
    system: {
      shouldUseNSFWChecker: false,
      shouldUseWatermarker: false,
    },
  } as never,
});

/** Find the flux2_klein_model_loader node in the graph. */
const getLoaderNode = async () => {
  const { g } = await buildFLUXGraph(buildGraphArg());
  const graph = g.getGraph();
  const loaderEntry = Object.entries(graph.nodes).find(([id]) => id.startsWith('flux2_klein_model_loader:'));
  return loaderEntry?.[1] as Record<string, unknown> | undefined;
};

describe('buildFLUXGraph – FLUX.2 Klein qwen3_source_model', () => {
  afterEach(() => {
    nextId = 0;
    model = { ...flux2DiffusersModel };
    kleinVaeModel = null;
    kleinQwen3EncoderModel = null;
    diffusersModels = [];
  });

  it('does not set qwen3_source_model when main model is diffusers', async () => {
    model = { ...flux2DiffusersModel };
    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.qwen3_source_model).toBeUndefined();
  });

  it('sets qwen3_source_model when main model is GGUF and a diffusers model is available', async () => {
    model = { ...flux2GGUFModel };
    diffusersModels = [diffusersSourceModelFixture];

    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.qwen3_source_model).toEqual({
      key: diffusersSourceModelFixture.key,
      hash: diffusersSourceModelFixture.hash,
      name: diffusersSourceModelFixture.name,
      base: diffusersSourceModelFixture.base,
      type: diffusersSourceModelFixture.type,
    });
  });

  it('does not set qwen3_source_model when main model is GGUF but standalone VAE and Qwen3 are both selected', async () => {
    model = { ...flux2GGUFModel };
    kleinVaeModel = kleinVaeModelFixture;
    kleinQwen3EncoderModel = kleinQwen3EncoderModelFixture;
    diffusersModels = [diffusersSourceModelFixture];

    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.qwen3_source_model).toBeUndefined();
  });

  it('does not set qwen3_source_model when main model is GGUF and no diffusers model is available', async () => {
    model = { ...flux2GGUFModel };
    diffusersModels = [];

    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.qwen3_source_model).toBeUndefined();
  });

  it('sets qwen3_source_model when only VAE is selected but Qwen3 is missing', async () => {
    model = { ...flux2GGUFModel };
    kleinVaeModel = kleinVaeModelFixture;
    kleinQwen3EncoderModel = null;
    diffusersModels = [diffusersSourceModelFixture];

    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.qwen3_source_model).toBeDefined();
  });

  it('sets qwen3_source_model when only Qwen3 is selected but VAE is missing', async () => {
    model = { ...flux2GGUFModel };
    kleinVaeModel = null;
    kleinQwen3EncoderModel = kleinQwen3EncoderModelFixture;
    diffusersModels = [diffusersSourceModelFixture];

    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.qwen3_source_model).toBeDefined();
  });

  it('passes standalone vae_model and qwen3_encoder_model when selected', async () => {
    model = { ...flux2DiffusersModel };
    kleinVaeModel = kleinVaeModelFixture;
    kleinQwen3EncoderModel = kleinQwen3EncoderModelFixture;

    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.vae_model).toEqual(kleinVaeModelFixture);
    expect(loader!.qwen3_encoder_model).toEqual(kleinQwen3EncoderModelFixture);
    expect(loader!.qwen3_source_model).toBeUndefined();
  });

  describe('variant matching', () => {
    it('selects a variant-matching diffusers model when multiple are available', async () => {
      model = { ...flux2GGUF9BModel };
      diffusersModels = [diffusersSourceModelFixture, diffusers9BSourceModelFixture];

      const loader = await getLoaderNode();
      expect(loader).toBeDefined();
      // Should pick the 9B diffusers model, not the 4B
      expect(loader!.qwen3_source_model).toEqual(expect.objectContaining({ key: diffusers9BSourceModelFixture.key }));
    });

    it('falls back to any diffusers model for VAE when standalone Qwen3 is selected but no variant match', async () => {
      model = { ...flux2GGUF9BModel };
      kleinQwen3EncoderModel = kleinQwen3EncoderModelFixture;
      // Only 4B diffusers available, no 9B — but Qwen3 is already provided standalone
      diffusersModels = [diffusersSourceModelFixture];

      const loader = await getLoaderNode();
      expect(loader).toBeDefined();
      // Should use the 4B diffusers model just for VAE extraction
      expect(loader!.qwen3_source_model).toEqual(expect.objectContaining({ key: diffusersSourceModelFixture.key }));
    });

    it('does not set qwen3_source_model when GGUF 9B with only 4B diffusers available and no standalone Qwen3', async () => {
      model = { ...flux2GGUF9BModel };
      kleinQwen3EncoderModel = null;
      // Only 4B diffusers available — wrong variant for Qwen3, no standalone Qwen3 selected
      diffusersModels = [diffusersSourceModelFixture];

      const loader = await getLoaderNode();
      expect(loader).toBeDefined();
      // Should NOT use the 4B diffusers since it has the wrong Qwen3 encoder
      expect(loader!.qwen3_source_model).toBeUndefined();
    });
  });

  describe('graph structure', () => {
    it('uses flux2_klein_model_loader for flux2 models', async () => {
      model = { ...flux2DiffusersModel };
      const { g } = await buildFLUXGraph(buildGraphArg());
      const graph = g.getGraph();
      const nodeIds = Object.keys(graph.nodes);
      expect(nodeIds.some((id) => id.startsWith('flux2_klein_model_loader:'))).toBe(true);
    });

    it('uses flux2_vae_decode for flux2 models', async () => {
      model = { ...flux2DiffusersModel };
      const { g } = await buildFLUXGraph(buildGraphArg());
      const graph = g.getGraph();
      const nodeIds = Object.keys(graph.nodes);
      expect(nodeIds.some((id) => id.startsWith('flux2_vae_decode:'))).toBe(true);
    });

    it('uses flux2_klein_text_encoder for flux2 models', async () => {
      model = { ...flux2DiffusersModel };
      const { g } = await buildFLUXGraph(buildGraphArg());
      const graph = g.getGraph();
      const nodeIds = Object.keys(graph.nodes);
      expect(nodeIds.some((id) => id.startsWith('flux2_klein_text_encoder:'))).toBe(true);
    });

    it('uses flux2_denoise for flux2 models', async () => {
      model = { ...flux2DiffusersModel };
      const { g } = await buildFLUXGraph(buildGraphArg());
      const graph = g.getGraph();
      const nodeTypes = Object.values(graph.nodes).map((n) => n.type);
      expect(nodeTypes).toContain('flux2_denoise');
    });
  });
});
