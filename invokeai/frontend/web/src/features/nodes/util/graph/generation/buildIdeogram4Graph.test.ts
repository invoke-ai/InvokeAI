import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('app/logging/logger', () => ({
  logger: () => ({ debug: vi.fn() }),
}));

let nextId = 0;
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:${nextId++}`,
}));

const model = { key: 'ideogram4-model', hash: 'ideogram4-hash', name: 'Ideogram 4', base: 'ideogram-4', type: 'main' };

// Controlled prompt inputs — the structured case (regions + palette) is the one that used to be
// clobbered by the decoy. globalPrompt is what the linear batch injects into.
let promptInputs = {
  globalPrompt: 'a global prompt',
  regions: [{ prompt: 'a red bird', bbox: [10, 20, 300, 400] as [number, number, number, number] }],
  colorPalette: ['#FF0000'],
};

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn(() => model),
  selectIdeogram4SamplerPreset: vi.fn(() => 'V4_QUALITY_48'),
  selectIdeogram4Steps: vi.fn(() => null),
  selectIdeogram4GuidanceScale: vi.fn(() => null),
  selectIdeogram4Mu: vi.fn(() => null),
}));

vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasMetadata: vi.fn(() => ({})),
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

vi.mock('features/nodes/util/graph/generation/buildIdeogram4Prompt', () => ({
  collectIdeogram4PromptInputs: vi.fn(() => promptInputs),
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  getOriginalAndScaledSizesForTextToImage: vi.fn(() => ({
    originalSize: { width: 1024, height: 1024 },
    scaledSize: { width: 1024, height: 1024 },
  })),
  selectCanvasOutputFields: vi.fn(() => ({})),
}));

vi.mock('features/ui/store/uiSelectors', () => ({
  selectActiveTab: vi.fn(() => 'generation'),
}));

vi.mock('services/api/types', async () => {
  const actual = await vi.importActual('services/api/types');
  return { ...actual, isNonRefinerMainModelConfig: vi.fn(() => true) };
});

import { buildIdeogram4Graph } from './buildIdeogram4Graph';

const state = { system: { shouldUseNSFWChecker: false, shouldUseWatermarker: false } };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const buildArg = (): any => ({ generationMode: 'txt2img', state, manager: null });

describe('buildIdeogram4Graph', () => {
  afterEach(() => {
    nextId = 0;
    promptInputs = {
      globalPrompt: 'a global prompt',
      regions: [{ prompt: 'a red bird', bbox: [10, 20, 300, 400] }],
      colorPalette: ['#FF0000'],
    };
  });

  it('routes the batch-injectable prompt node through the caption builder into the text encoder', async () => {
    const { g, positivePrompt } = await buildIdeogram4Graph(buildArg());

    // The returned positivePrompt (the node the linear batch / dynamic prompts inject into) is the real
    // prompt node — NOT a decoy — so injected expansions reach the encoder instead of a throwaway node.
    expect(positivePrompt.type).toBe('string');
    expect(positivePrompt.id).toContain('ideogram4_prompt');
    expect(positivePrompt.id).not.toContain('decoy');
    expect(positivePrompt.value).toBe('a global prompt');

    const nodes = g.getNodes();
    const captionBuilder = nodes.find((n) => n.type === 'ideogram4_caption_builder');
    const textEncoder = nodes.find((n) => n.type === 'ideogram4_text_encoder');
    expect(captionBuilder).toBeDefined();
    expect(textEncoder).toBeDefined();

    const edges = g.getEdges();
    // prompt node -> caption builder (so an injected prompt is what the caption is assembled from)
    expect(
      edges.some(
        (e) =>
          e.source.node_id === positivePrompt.id &&
          e.source.field === 'value' &&
          e.destination.node_id === captionBuilder!.id &&
          e.destination.field === 'prompt'
      )
    ).toBe(true);
    // caption builder -> text encoder (the assembled caption is what actually gets encoded)
    expect(
      edges.some(
        (e) =>
          e.source.node_id === captionBuilder!.id &&
          e.source.field === 'value' &&
          e.destination.node_id === textEncoder!.id &&
          e.destination.field === 'prompt'
      )
    ).toBe(true);
  });

  it('bakes the fixed regions and palette into the caption builder node', async () => {
    const { g } = await buildIdeogram4Graph(buildArg());
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const captionBuilder = g.getNodes().find((n) => n.type === 'ideogram4_caption_builder') as any;
    expect(captionBuilder.regions).toEqual([{ prompt: 'a red bird', bbox: [10, 20, 300, 400] }]);
    expect(captionBuilder.color_palette).toEqual(['#FF0000']);
  });

  // `core_metadata` and the `ideogram4_caption` metadata field are outside Graph's strict typed unions
  // (the metadata node is excluded from AnyInvocation), so compare their string values via casts.
  const captionEdge = (g: Awaited<ReturnType<typeof buildIdeogram4Graph>>['g']) => {
    const captionBuilderId = g.getNodes().find((n) => (n.type as string) === 'ideogram4_caption_builder')?.id;
    return g
      .getEdges()
      .find((e) => e.source.node_id === captionBuilderId && (e.destination.field as string) === 'ideogram4_caption');
  };

  it('records the runtime-assembled caption in metadata via an edge (structured inputs)', async () => {
    const { g } = await buildIdeogram4Graph(buildArg());
    // The caption is wired from the runtime builder into metadata, so each batch item records its own.
    const edge = captionEdge(g);
    expect(edge).toBeDefined();
    expect(edge!.source.field).toBe('value');
  });

  it('always records the caption edge, even for a plain prompt (so the encoded JSON is visible)', async () => {
    // Plain prompts are now wrapped in the JSON schema at runtime, so the caption differs from the raw
    // positive_prompt and must be recorded too — otherwise the metadata viewer only shows the raw text.
    promptInputs = { globalPrompt: 'just plain text', regions: [], colorPalette: [] };
    const { g } = await buildIdeogram4Graph(buildArg());
    expect(captionEdge(g)).toBeDefined();
  });
});
