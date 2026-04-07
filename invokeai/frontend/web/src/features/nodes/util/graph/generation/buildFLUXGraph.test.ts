import { buildFLUXGraph } from 'features/nodes/util/graph/generation/buildFLUXGraph';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { GraphBuilderArg } from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// ---------------------------------------------------------------------------
// Module mocks
//
// `buildFLUXGraph` pulls in a large slice of the app: redux selectors, every
// `add*` helper, validators, the canvas manager, etc. The function itself only
// orchestrates these; the unit under test here is the orchestration logic
// (variant-gated guidance, scheduler propagation, metadata persistence). So we
// stub out every collaborator and assert against the resulting `Graph` object.
// ---------------------------------------------------------------------------

const mockState = {
  // buildFLUXGraph reads `state.system.shouldUse{NSFWChecker,Watermarker}` directly,
  // every other access is funneled through the mocked selectors below.
  system: { shouldUseNSFWChecker: false, shouldUseWatermarker: false },
} as unknown as Parameters<typeof buildFLUXGraph>[0]['state'];

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

let currentModel: { key: string; hash: string; name: string; base: string; type: string; variant?: string } | null;
let currentKleinVae: { key: string; hash: string; name: string; base: string; type: string } | null;
let currentKleinQwen3: { key: string; hash: string; name: string; base: string; type: string } | null;

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: () => currentModel,
  selectParamsSlice: () => mockParams,
  selectKleinVaeModel: () => currentKleinVae,
  selectKleinQwen3EncoderModel: () => currentKleinQwen3,
}));

vi.mock('features/controlLayers/store/refImagesSlice', () => ({
  selectRefImagesSlice: () => ({ entities: [] }),
}));

vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasSlice: () => ({
    bbox: { rect: { x: 0, y: 0, width: 1024, height: 1024 } },
    controlLayers: { entities: [] },
    regionalGuidance: { entities: [] },
  }),
  selectCanvasMetadata: () => ({}),
}));

vi.mock('features/controlLayers/store/types', () => ({
  isFlux2ReferenceImageConfig: () => false,
  isFluxKontextReferenceImageConfig: () => false,
}));

vi.mock('features/controlLayers/store/validators', () => ({
  getGlobalReferenceImageWarnings: () => [],
}));

vi.mock('features/ui/store/uiSelectors', () => ({
  selectActiveTab: () => 'generate',
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  selectCanvasOutputFields: () => ({}),
}));

// Helper add* functions: each test cares only that the FLUX.2 orchestration
// path produces the right metadata + denoise inputs. The actual node graph
// produced by these helpers is irrelevant here.
vi.mock('features/nodes/util/graph/generation/addTextToImage', () => ({
  addTextToImage: ({ l2i }: { l2i: Invocation<'flux2_vae_decode'> }) => l2i,
}));
vi.mock('features/nodes/util/graph/generation/addImageToImage', () => ({
  addImageToImage: vi.fn(),
}));
vi.mock('features/nodes/util/graph/generation/addInpaint', () => ({ addInpaint: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addOutpaint', () => ({ addOutpaint: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addNSFWChecker', () => ({ addNSFWChecker: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addWatermarker', () => ({ addWatermarker: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addRegions', () => ({ addRegions: vi.fn(() => []) }));
vi.mock('features/nodes/util/graph/generation/addFLUXLoRAs', () => ({ addFLUXLoRAs: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addFlux2KleinLoRAs', () => ({ addFlux2KleinLoRAs: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addFLUXFill', () => ({ addFLUXFill: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addFLUXRedux', () => ({
  addFLUXReduxes: () => ({ addedFLUXReduxes: 0 }),
}));
vi.mock('features/nodes/util/graph/generation/addControlAdapters', () => ({
  addControlNets: vi.fn(() => Promise.resolve({ addedControlNets: 0 })),
  addControlLoRA: vi.fn(),
}));
vi.mock('features/nodes/util/graph/generation/addIPAdapters', () => ({
  addIPAdapters: () => ({ addedIPAdapters: 0 }),
}));

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

const makeFlux2Model = (variant: string) => ({
  key: `flux2-${variant}`,
  hash: 'hash',
  name: `FLUX.2 Klein ${variant}`,
  base: 'flux2',
  type: 'main',
  variant,
});

const buildArg = (): GraphBuilderArg =>
  ({
    generationMode: 'txt2img',
    state: mockState,
    manager: null,
  }) as unknown as GraphBuilderArg;

const findFlux2Denoise = (g: Graph): Invocation<'flux2_denoise'> | undefined => {
  // The Graph object stores nodes on `_graph.nodes` keyed by id.
  const nodes = (g as unknown as { _graph: { nodes: Record<string, { type: string }> } })._graph.nodes;
  return Object.values(nodes).find((n) => n.type === 'flux2_denoise') as Invocation<'flux2_denoise'> | undefined;
};

const getMetadata = (g: Graph): Record<string, unknown> =>
  (g as unknown as { getMetadataNode: () => Record<string, unknown> }).getMetadataNode();

beforeEach(() => {
  currentModel = null;
  currentKleinVae = null;
  currentKleinQwen3 = null;
});

describe('buildFLUXGraph (FLUX.2 Klein)', () => {
  describe('guidance gating', () => {
    it('writes guidance into metadata and the denoise node for klein_9b_base', async () => {
      currentModel = makeFlux2Model('klein_9b_base');

      const { g } = await buildFLUXGraph(buildArg());

      const metadata = getMetadata(g);
      expect(metadata.guidance).toBe(mockParams.guidance);

      const denoise = findFlux2Denoise(g);
      expect(denoise).toBeDefined();
      expect(denoise?.guidance).toBe(mockParams.guidance);
    });

    it.each(['klein_9b', 'klein_4b'])(
      'omits guidance from metadata and denoise for distilled variant %s',
      async (variant) => {
        currentModel = makeFlux2Model(variant);

        const { g } = await buildFLUXGraph(buildArg());

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

      const { g } = await buildFLUXGraph(buildArg());

      expect(getMetadata(g).scheduler).toBe(mockParams.fluxScheduler);
      expect(findFlux2Denoise(g)?.scheduler).toBe(mockParams.fluxScheduler);
    });
  });

  describe('Klein VAE / Qwen3 metadata', () => {
    it('persists separately selected Klein VAE and Qwen3 encoder into metadata', async () => {
      currentModel = makeFlux2Model('klein_9b_base');
      currentKleinVae = { key: 'vae-1', hash: 'h', name: 'Klein VAE', base: 'flux2', type: 'vae' };
      currentKleinQwen3 = { key: 'q3-1', hash: 'h', name: 'Qwen3', base: 'flux2', type: 'qwen3_encoder' };

      const { g } = await buildFLUXGraph(buildArg());

      const metadata = getMetadata(g);
      expect(metadata.vae).toEqual(currentKleinVae);
      expect(metadata.qwen3_encoder).toEqual(currentKleinQwen3);
    });

    it('omits vae / qwen3_encoder when none are selected', async () => {
      currentModel = makeFlux2Model('klein_9b_base');

      const { g } = await buildFLUXGraph(buildArg());

      const metadata = getMetadata(g);
      expect(metadata.vae).toBeUndefined();
      expect(metadata.qwen3_encoder).toBeUndefined();
    });
  });
});
