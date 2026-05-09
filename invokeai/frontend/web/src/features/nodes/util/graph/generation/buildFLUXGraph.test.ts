import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// ---------------------------------------------------------------------------
// Module mocks
//
// `buildFLUXGraph` pulls in a large slice of the app: redux selectors, every
// `add*` helper, validators, the canvas manager, etc. The function itself only
// orchestrates these; the units under test here are the orchestration bits
// (variant-gated guidance, scheduler propagation, metadata persistence, and
// qwen3_source_model auto-detection for GGUF Klein). So we stub out every
// collaborator and assert against the resulting `Graph` object.
// ---------------------------------------------------------------------------

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

const makeFlux2Model = (variant: string) => ({
  key: `flux2-${variant}`,
  hash: 'hash',
  name: `FLUX.2 Klein ${variant}`,
  base: 'flux2',
  type: 'main',
  format: 'diffusers',
  variant,
});

// --- Mutable state shared by all tests ---

let currentModel: Record<string, unknown> | null = null;
let currentKleinVae: Record<string, unknown> | null = null;
let currentKleinQwen3: Record<string, unknown> | null = null;
let diffusersModels: Record<string, unknown>[] = [];

const mockParams = {
  guidance: 3.5,
  steps: 28,
  fluxScheduler: 'euler' as const,
  fluxDypePreset: 'off' as const,
  fluxDypeScale: 1,
  fluxDypeExponent: 1,
  fluxVAE: null,
  t5EncoderModel: null,
  clipEmbedModel: null,
};

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn(() => currentModel),
  selectParamsSlice: vi.fn(() => mockParams),
  selectKleinVaeModel: vi.fn(() => currentKleinVae),
  selectKleinQwen3EncoderModel: vi.fn(() => currentKleinQwen3),
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

vi.mock('features/controlLayers/store/types', () => ({
  isFlux2ReferenceImageConfig: vi.fn(() => false),
  isFluxKontextReferenceImageConfig: vi.fn(() => false),
}));

vi.mock('features/controlLayers/store/validators', () => ({
  getGlobalReferenceImageWarnings: vi.fn(() => []),
}));

vi.mock('features/ui/store/uiSelectors', () => ({
  selectActiveTab: vi.fn(() => 'generate'),
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  selectCanvasOutputFields: vi.fn(() => ({})),
  selectPresetModifiedPrompts: vi.fn(() => ({
    positive: 'a prompt',
    negative: '',
  })),
}));

// Helper add* functions: the tests care about the FLUX.2 orchestration path
// (metadata, denoise inputs, loader inputs). The actual node graphs produced
// by these helpers are irrelevant here.
vi.mock('features/nodes/util/graph/generation/addTextToImage', () => ({
  addTextToImage: vi.fn(({ l2i }) => l2i),
}));
vi.mock('features/nodes/util/graph/generation/addImageToImage', () => ({
  addImageToImage: vi.fn(),
}));
vi.mock('features/nodes/util/graph/generation/addInpaint', () => ({ addInpaint: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addOutpaint', () => ({ addOutpaint: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addNSFWChecker', () => ({
  addNSFWChecker: vi.fn((_g, node) => node),
}));
vi.mock('features/nodes/util/graph/generation/addWatermarker', () => ({
  addWatermarker: vi.fn((_g, node) => node),
}));
vi.mock('features/nodes/util/graph/generation/addRegions', () => ({ addRegions: vi.fn(() => []) }));
vi.mock('features/nodes/util/graph/generation/addFLUXLoRAs', () => ({ addFLUXLoRAs: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addFlux2KleinLoRAs', () => ({ addFlux2KleinLoRAs: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addFLUXFill', () => ({ addFLUXFill: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addFLUXRedux', () => ({
  addFLUXReduxes: vi.fn(() => ({ addedFLUXReduxes: 0 })),
}));
vi.mock('features/nodes/util/graph/generation/addControlAdapters', () => ({
  addControlNets: vi.fn(() => Promise.resolve({ addedControlNets: 0 })),
  addControlLoRA: vi.fn(),
}));
vi.mock('features/nodes/util/graph/generation/addIPAdapters', () => ({
  addIPAdapters: vi.fn(() => ({ addedIPAdapters: 0 })),
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

import type { GraphBuilderArg } from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';

import { buildFLUXGraph } from './buildFLUXGraph';
import type { Graph } from './Graph';

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

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

const findFlux2Denoise = (g: Graph): Invocation<'flux2_denoise'> | undefined => {
  const nodes = (g as unknown as { _graph: { nodes: Record<string, { type: string }> } })._graph.nodes;
  return Object.values(nodes).find((n) => n.type === 'flux2_denoise') as Invocation<'flux2_denoise'> | undefined;
};

const getMetadata = (g: Graph): Record<string, unknown> =>
  (g as unknown as { getMetadataNode: () => Record<string, unknown> }).getMetadataNode();

const getLoaderNode = async () => {
  const { g } = await buildFLUXGraph(buildGraphArg());
  const graph = g.getGraph();
  const loaderEntry = Object.entries(graph.nodes).find(([id]) => id.startsWith('flux2_klein_model_loader:'));
  return loaderEntry?.[1] as Record<string, unknown> | undefined;
};

const resetState = () => {
  nextId = 0;
  currentModel = null;
  currentKleinVae = null;
  currentKleinQwen3 = null;
  diffusersModels = [];
};

beforeEach(resetState);
afterEach(resetState);

describe('buildFLUXGraph (FLUX.2 Klein)', () => {
  describe('guidance gating', () => {
    // guidance_embeds is inert for all current FLUX.2 Klein variants (weights are
    // absent or zeroed), so the linear UI does not expose it and the graph builder
    // must not write it into the denoise node or metadata.
    it.each(['klein_9b_base', 'klein_9b', 'klein_4b_base', 'klein_4b'])(
      'omits guidance from metadata and denoise for variant %s',
      async (variant) => {
        currentModel = makeFlux2Model(variant);

        const { g } = await buildFLUXGraph(buildGraphArg());

        const metadata = getMetadata(g);
        expect(metadata.guidance).toBeUndefined();

        const denoise = findFlux2Denoise(g);
        expect(denoise).toBeDefined();
        expect(denoise?.guidance).toBeUndefined();
      }
    );
  });

  describe('scheduler persistence', () => {
    it('writes the FLUX scheduler into metadata and the denoise node for FLUX.2', async () => {
      currentModel = makeFlux2Model('klein_9b_base');

      const { g } = await buildFLUXGraph(buildGraphArg());

      expect(getMetadata(g).scheduler).toBe(mockParams.fluxScheduler);
      expect(findFlux2Denoise(g)?.scheduler).toBe(mockParams.fluxScheduler);
    });
  });

  describe('Klein VAE / Qwen3 metadata', () => {
    it('persists separately selected Klein VAE and Qwen3 encoder into metadata', async () => {
      currentModel = makeFlux2Model('klein_9b_base');
      currentKleinVae = { key: 'vae-1', hash: 'h', name: 'Klein VAE', base: 'flux2', type: 'vae' };
      currentKleinQwen3 = { key: 'q3-1', hash: 'h', name: 'Qwen3', base: 'flux2', type: 'qwen3_encoder' };

      const { g } = await buildFLUXGraph(buildGraphArg());

      const metadata = getMetadata(g);
      expect(metadata.vae).toEqual(currentKleinVae);
      expect(metadata.qwen3_encoder).toEqual(currentKleinQwen3);
    });

    it('omits vae / qwen3_encoder when none are selected', async () => {
      currentModel = makeFlux2Model('klein_9b_base');

      const { g } = await buildFLUXGraph(buildGraphArg());

      const metadata = getMetadata(g);
      expect(metadata.vae).toBeUndefined();
      expect(metadata.qwen3_encoder).toBeUndefined();
    });
  });
});

describe('buildFLUXGraph – FLUX.2 Klein qwen3_source_model', () => {
  it('does not set qwen3_source_model when main model is diffusers', async () => {
    currentModel = { ...flux2DiffusersModel };
    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.qwen3_source_model).toBeUndefined();
  });

  it('sets qwen3_source_model when main model is GGUF and a diffusers model is available', async () => {
    currentModel = { ...flux2GGUFModel };
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
    currentModel = { ...flux2GGUFModel };
    currentKleinVae = kleinVaeModelFixture;
    currentKleinQwen3 = kleinQwen3EncoderModelFixture;
    diffusersModels = [diffusersSourceModelFixture];

    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.qwen3_source_model).toBeUndefined();
  });

  it('does not set qwen3_source_model when main model is GGUF and no diffusers model is available', async () => {
    currentModel = { ...flux2GGUFModel };
    diffusersModels = [];

    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.qwen3_source_model).toBeUndefined();
  });

  it('sets qwen3_source_model when only VAE is selected but Qwen3 is missing', async () => {
    currentModel = { ...flux2GGUFModel };
    currentKleinVae = kleinVaeModelFixture;
    currentKleinQwen3 = null;
    diffusersModels = [diffusersSourceModelFixture];

    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.qwen3_source_model).toBeDefined();
  });

  it('sets qwen3_source_model when only Qwen3 is selected but VAE is missing', async () => {
    currentModel = { ...flux2GGUFModel };
    currentKleinVae = null;
    currentKleinQwen3 = kleinQwen3EncoderModelFixture;
    diffusersModels = [diffusersSourceModelFixture];

    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.qwen3_source_model).toBeDefined();
  });

  it('passes standalone vae_model and qwen3_encoder_model when selected', async () => {
    currentModel = { ...flux2DiffusersModel };
    currentKleinVae = kleinVaeModelFixture;
    currentKleinQwen3 = kleinQwen3EncoderModelFixture;

    const loader = await getLoaderNode();
    expect(loader).toBeDefined();
    expect(loader!.vae_model).toEqual(kleinVaeModelFixture);
    expect(loader!.qwen3_encoder_model).toEqual(kleinQwen3EncoderModelFixture);
    expect(loader!.qwen3_source_model).toBeUndefined();
  });

  describe('variant matching', () => {
    it('selects a variant-matching diffusers model when multiple are available', async () => {
      currentModel = { ...flux2GGUF9BModel };
      diffusersModels = [diffusersSourceModelFixture, diffusers9BSourceModelFixture];

      const loader = await getLoaderNode();
      expect(loader).toBeDefined();
      expect(loader!.qwen3_source_model).toEqual(expect.objectContaining({ key: diffusers9BSourceModelFixture.key }));
    });

    it('falls back to any diffusers model for VAE when standalone Qwen3 is selected but no variant match', async () => {
      currentModel = { ...flux2GGUF9BModel };
      currentKleinQwen3 = kleinQwen3EncoderModelFixture;
      diffusersModels = [diffusersSourceModelFixture];

      const loader = await getLoaderNode();
      expect(loader).toBeDefined();
      expect(loader!.qwen3_source_model).toEqual(expect.objectContaining({ key: diffusersSourceModelFixture.key }));
    });

    it('does not set qwen3_source_model when GGUF 9B with only 4B diffusers available and no standalone Qwen3', async () => {
      currentModel = { ...flux2GGUF9BModel };
      currentKleinQwen3 = null;
      diffusersModels = [diffusersSourceModelFixture];

      const loader = await getLoaderNode();
      expect(loader).toBeDefined();
      expect(loader!.qwen3_source_model).toBeUndefined();
    });
  });

  describe('graph structure', () => {
    it('uses flux2_klein_model_loader for flux2 models', async () => {
      currentModel = { ...flux2DiffusersModel };
      const { g } = await buildFLUXGraph(buildGraphArg());
      const graph = g.getGraph();
      const nodeIds = Object.keys(graph.nodes);
      expect(nodeIds.some((id) => id.startsWith('flux2_klein_model_loader:'))).toBe(true);
    });

    it('uses flux2_vae_decode for flux2 models', async () => {
      currentModel = { ...flux2DiffusersModel };
      const { g } = await buildFLUXGraph(buildGraphArg());
      const graph = g.getGraph();
      const nodeIds = Object.keys(graph.nodes);
      expect(nodeIds.some((id) => id.startsWith('flux2_vae_decode:'))).toBe(true);
    });

    it('uses flux2_klein_text_encoder for flux2 models', async () => {
      currentModel = { ...flux2DiffusersModel };
      const { g } = await buildFLUXGraph(buildGraphArg());
      const graph = g.getGraph();
      const nodeIds = Object.keys(graph.nodes);
      expect(nodeIds.some((id) => id.startsWith('flux2_klein_text_encoder:'))).toBe(true);
    });

    it('uses flux2_denoise for flux2 models', async () => {
      currentModel = { ...flux2DiffusersModel };
      const { g } = await buildFLUXGraph(buildGraphArg());
      const graph = g.getGraph();
      const nodeTypes = Object.values(graph.nodes).map((n) => n.type);
      expect(nodeTypes).toContain('flux2_denoise');
    });
  });
});
