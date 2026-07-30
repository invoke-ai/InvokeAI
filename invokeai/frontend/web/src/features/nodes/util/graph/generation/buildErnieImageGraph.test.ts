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

const model = {
  key: 'ernie-model',
  hash: 'ernie-hash',
  name: 'ERNIE-Image',
  base: 'ernie-image',
  type: 'main',
  format: 'diffusers',
};

const defaultParams = {
  cfgScale: 4 as number | number[],
  steps: 50,
};

let params = { ...defaultParams };
let usePromptEnhancer = true;

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn(() => model),
  selectParamsSlice: vi.fn(() => params),
  selectErnieImageScheduler: vi.fn(() => 'euler'),
  selectErnieImageUsePromptEnhancer: vi.fn(() => usePromptEnhancer),
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

// Stand in for the real `addTextToImage`, which is what resolves the generation dimensions and
// writes them onto the denoise node. The size it reports is what the enhancer must end up with.
let scaledSize = { width: 832, height: 1216 };
vi.mock('features/nodes/util/graph/generation/addTextToImage', () => ({
  addTextToImage: vi.fn(({ denoise, l2i }) => {
    denoise.width = scaledSize.width;
    denoise.height = scaledSize.height;
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

import { buildErnieImageGraph } from './buildErnieImageGraph';

type BuiltGraph = Awaited<ReturnType<typeof buildErnieImageGraph>>['g'];

const build = (generationMode: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint' = 'txt2img') =>
  buildErnieImageGraph({
    generationMode,
    manager: null,
    state: {
      system: { shouldUseNSFWChecker: false, shouldUseWatermarker: false },
    } as never,
  });

const nodesOf = (g: BuiltGraph) => Object.values(g.getGraph().nodes) as Record<string, unknown>[];
const positiveEncoder = (g: BuiltGraph) =>
  nodesOf(g).find((n) => n.type === 'ernie_image_text_encoder' && String(n.id).startsWith('pos_prompt'));

describe('buildErnieImageGraph', () => {
  beforeEach(() => {
    nextId = 0;
    params = { ...defaultParams };
    usePromptEnhancer = true;
    scaledSize = { width: 832, height: 1216 };
  });

  it('builds the core txt2img node chain', async () => {
    const { g } = await build();
    const types = nodesOf(g).map((n) => n.type);
    expect(types).toContain('ernie_image_model_loader');
    expect(types).toContain('ernie_image_text_encoder');
    expect(types).toContain('ernie_image_denoise');
    expect(types).toContain('ernie_image_vae_decode');
  });

  it.each(['img2img', 'inpaint', 'outpaint'] as const)('refuses the %s generation mode', async (mode) => {
    // ERNIE has no VAE-encode node and no denoise_mask input, so these modes cannot be built.
    // The dedicated error type is what surfaces the friendly toast instead of a red graph error.
    await expect(build(mode)).rejects.toThrow(/only supports text-to-image/);
  });

  describe('prompt enhancer', () => {
    // The enhancer is handed the target size and rewrites the prompt to suit that aspect ratio.
    // Its node defaults are 1024x1024, so leaving `pe_width`/`pe_height` unwired means a portrait
    // generation is enhanced for a square image.
    it('passes the real generation dimensions to the enhancer', async () => {
      const { g } = await build();

      expect(positiveEncoder(g)).toMatchObject({ pe_width: 832, pe_height: 1216 });
    });

    it('tracks the generation dimensions rather than hardcoding them', async () => {
      scaledSize = { width: 1536, height: 640 };
      const { g } = await build();

      expect(positiveEncoder(g)).toMatchObject({ pe_width: 1536, pe_height: 640 });
    });

    it('wires the enhancer edge only when the toggle is on', async () => {
      const { g } = await build();
      expect(g.getGraph().edges.some((e) => e.destination.field === 'prompt_enhancer')).toBe(true);

      nextId = 0;
      usePromptEnhancer = false;
      const { g: gOff } = await build();
      expect(gOff.getGraph().edges.some((e) => e.destination.field === 'prompt_enhancer')).toBe(false);
    });

    it('never enhances the negative prompt', async () => {
      const { g } = await build();
      const negEncoder = nodesOf(g).find(
        (n) => n.type === 'ernie_image_text_encoder' && String(n.id).startsWith('neg_prompt')
      );

      expect(negEncoder).toMatchObject({ use_prompt_enhancer: false });
    });
  });

  describe('CFG gating (negative conditioning)', () => {
    it('omits the negative encoder + edge when cfg_scale <= 1 (Turbo default)', async () => {
      params = { ...defaultParams, cfgScale: 1 };
      const { g } = await build();

      expect(Object.keys(g.getGraph().nodes).some((id) => id.startsWith('neg_prompt:'))).toBe(false);
      expect(g.getGraph().edges.some((e) => e.destination.field === 'negative_conditioning')).toBe(false);
    });

    it('includes the negative encoder + edge when cfg_scale > 1', async () => {
      const { g } = await build();

      expect(g.getGraph().edges.some((e) => e.destination.field === 'negative_conditioning')).toBe(true);
    });
  });
});
