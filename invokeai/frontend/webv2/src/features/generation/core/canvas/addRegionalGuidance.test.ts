import { describe, expect, it } from 'vitest';

import type { AddRegionalGuidanceOptions, RegionalGuidanceInput, RegionalReferenceModel } from './addRegionalGuidance';

import {
  addRegionalGuidance,
  getRegionalGuidanceRejectionReason,
  isRegionalGuidanceSupportedForBase,
} from './addRegionalGuidance';

interface TestGraph {
  id: string;
  nodes: Record<string, { id: string; type: string; [key: string]: unknown }>;
  edges: { source: { node_id: string; field: string }; destination: { node_id: string; field: string } }[];
}

/** A minimal SD-family base graph: pos/neg cond + collectors wired to denoise, with clip edges. */
const sdBaseGraph = (opts: { sdxl?: boolean } = {}): TestGraph => {
  const clip2Edges = opts.sdxl
    ? [
        { destination: { field: 'clip2', node_id: 'pos_cond' }, source: { field: 'clip2', node_id: 'model_loader' } },
        { destination: { field: 'clip2', node_id: 'neg_cond' }, source: { field: 'clip2', node_id: 'model_loader' } },
      ]
    : [];
  return {
    edges: [
      { destination: { field: 'clip', node_id: 'pos_cond' }, source: { field: 'clip', node_id: 'clip_skip' } },
      { destination: { field: 'clip', node_id: 'neg_cond' }, source: { field: 'clip', node_id: 'clip_skip' } },
      ...clip2Edges,
      {
        destination: { field: 'positive_conditioning', node_id: 'denoise_latents' },
        source: { field: 'collection', node_id: 'pos_cond_collect' },
      },
      {
        destination: { field: 'negative_conditioning', node_id: 'denoise_latents' },
        source: { field: 'collection', node_id: 'neg_cond_collect' },
      },
    ],
    id: 'g',
    nodes: {
      denoise_latents: { id: 'denoise_latents', type: 'denoise_latents' },
      neg_cond: { id: 'neg_cond', type: opts.sdxl ? 'sdxl_compel_prompt' : 'compel' },
      neg_cond_collect: { id: 'neg_cond_collect', type: 'collect' },
      pos_cond: { id: 'pos_cond', type: opts.sdxl ? 'sdxl_compel_prompt' : 'compel' },
      pos_cond_collect: { id: 'pos_cond_collect', type: 'collect' },
    },
  };
};

/** A minimal FLUX base graph: pos cond + collector, no negative. */
const fluxBaseGraph = (): TestGraph => ({
  edges: [
    { destination: { field: 'clip', node_id: 'pos_cond' }, source: { field: 'clip', node_id: 'model_loader' } },
    {
      destination: { field: 't5_encoder', node_id: 'pos_cond' },
      source: { field: 't5_encoder', node_id: 'model_loader' },
    },
    {
      destination: { field: 'positive_text_conditioning', node_id: 'denoise_latents' },
      source: { field: 'collection', node_id: 'pos_cond_collect' },
    },
  ],
  id: 'g',
  nodes: {
    denoise_latents: { id: 'denoise_latents', type: 'flux_denoise' },
    pos_cond: { id: 'pos_cond', type: 'flux_text_encoder' },
    pos_cond_collect: { id: 'pos_cond_collect', type: 'collect' },
  },
});

const ipModel = (base: string): RegionalReferenceModel => ({
  base,
  key: `ip-${base}`,
  name: `IP ${base}`,
  type: 'ip_adapter',
});

const region = (overrides: Partial<RegionalGuidanceInput> = {}): RegionalGuidanceInput => ({
  autoNegative: false,
  id: 'r1',
  maskImageName: 'mask1.png',
  negativePrompt: null,
  positivePrompt: 'a cat',
  referenceImages: [],
  ...overrides,
});

const run = (graph: TestGraph, options: AddRegionalGuidanceOptions): TestGraph => {
  addRegionalGuidance(graph as never, options);
  return graph;
};

const nodesOfType = (graph: TestGraph, type: string) => Object.values(graph.nodes).filter((n) => n.type === type);
const hasEdge = (graph: TestGraph, s: string, sf: string, d: string, df: string): boolean =>
  graph.edges.some(
    (e) => e.source.node_id === s && e.source.field === sf && e.destination.node_id === d && e.destination.field === df
  );

describe('isRegionalGuidanceSupportedForBase', () => {
  it('supports sd-1 / sdxl / flux and nothing else', () => {
    expect(isRegionalGuidanceSupportedForBase('sd-1')).toBe(true);
    expect(isRegionalGuidanceSupportedForBase('sdxl')).toBe(true);
    expect(isRegionalGuidanceSupportedForBase('flux')).toBe(true);
    expect(isRegionalGuidanceSupportedForBase('sd-3')).toBe(false);
    expect(isRegionalGuidanceSupportedForBase('cogview4')).toBe(false);
  });
});

describe('addRegionalGuidance — SD1', () => {
  it('wires a mask tensor + positive conditioning into the positive collector, copying clip', () => {
    const g = run(sdBaseGraph(), { base: 'sd-1', regions: [region()] });
    const mask = nodesOfType(g, 'alpha_mask_to_tensor');
    expect(mask).toHaveLength(1);
    expect(mask[0].image).toEqual({ image_name: 'mask1.png' });
    expect(mask[0].id).toBe('rg_mask_to_tensor_r1');

    const posCond = g.nodes.rg_pos_cond_r1;
    expect(posCond).toBeDefined();
    expect(posCond.type).toBe('compel');
    expect(posCond.prompt).toBe('a cat');
    // mask edge + conditioning into the collector.
    expect(hasEdge(g, 'rg_mask_to_tensor_r1', 'mask', 'rg_pos_cond_r1', 'mask')).toBe(true);
    expect(hasEdge(g, 'rg_pos_cond_r1', 'conditioning', 'pos_cond_collect', 'item')).toBe(true);
    // Copied the global posCond clip edge.
    expect(hasEdge(g, 'clip_skip', 'clip', 'rg_pos_cond_r1', 'clip')).toBe(true);
  });

  it('wires a negative prompt into the negative collector', () => {
    const g = run(sdBaseGraph(), { base: 'sd-1', regions: [region({ negativePrompt: 'blurry' })] });
    expect(g.nodes.rg_neg_cond_r1.prompt).toBe('blurry');
    expect(hasEdge(g, 'rg_mask_to_tensor_r1', 'mask', 'rg_neg_cond_r1', 'mask')).toBe(true);
    expect(hasEdge(g, 'rg_neg_cond_r1', 'conditioning', 'neg_cond_collect', 'item')).toBe(true);
    expect(hasEdge(g, 'clip_skip', 'clip', 'rg_neg_cond_r1', 'clip')).toBe(true);
  });

  it('autoNegative inverts the mask and re-encodes the positive prompt into the negative collector', () => {
    const g = run(sdBaseGraph(), { base: 'sd-1', regions: [region({ autoNegative: true })] });
    const invert = nodesOfType(g, 'invert_tensor_mask');
    expect(invert).toHaveLength(1);
    expect(invert[0].id).toBe('rg_invert_mask_r1');
    expect(hasEdge(g, 'rg_mask_to_tensor_r1', 'mask', 'rg_invert_mask_r1', 'mask')).toBe(true);
    // The inverted-cond node uses the POSITIVE prompt but feeds the NEGATIVE collector.
    expect(g.nodes.rg_pos_cond_inverted_r1.prompt).toBe('a cat');
    expect(hasEdge(g, 'rg_invert_mask_r1', 'mask', 'rg_pos_cond_inverted_r1', 'mask')).toBe(true);
    expect(hasEdge(g, 'rg_pos_cond_inverted_r1', 'conditioning', 'neg_cond_collect', 'item')).toBe(true);
  });

  it('does not autoNegative when there is no positive prompt', () => {
    const g = run(sdBaseGraph(), {
      base: 'sd-1',
      regions: [region({ autoNegative: true, negativePrompt: 'x', positivePrompt: null })],
    });
    expect(nodesOfType(g, 'invert_tensor_mask')).toHaveLength(0);
  });

  it('wires a regional ip_adapter reference image mask-scoped into a collector → denoise.ip_adapter', () => {
    const g = run(sdBaseGraph(), {
      base: 'sd-1',
      regions: [
        region({
          referenceImages: [
            {
              beginEndStepPct: [0, 1],
              clipVisionModel: 'ViT-H',
              id: 'ref1',
              imageName: 'ref.png',
              method: 'full',
              model: ipModel('sd-1'),
              type: 'ip_adapter',
              weight: 0.8,
            },
          ],
        }),
      ],
    });
    const ip = g.nodes.ip_adapter_ref1;
    expect(ip.type).toBe('ip_adapter');
    expect(ip.weight).toBe(0.8);
    expect(ip.image).toEqual({ image_name: 'ref.png' });
    expect(hasEdge(g, 'rg_mask_to_tensor_r1', 'mask', 'ip_adapter_ref1', 'mask')).toBe(true);
    // A collector feeds denoise.ip_adapter.
    const collectorEdge = g.edges.find(
      (e) => e.destination.node_id === 'denoise_latents' && e.destination.field === 'ip_adapter'
    );
    expect(collectorEdge).toBeDefined();
    expect(hasEdge(g, 'ip_adapter_ref1', 'ip_adapter', collectorEdge!.source.node_id, 'item')).toBe(true);
  });
});

describe('addRegionalGuidance — SDXL', () => {
  it('sets prompt + style on the sdxl encoder and copies clip + clip2', () => {
    const g = run(sdBaseGraph({ sdxl: true }), { base: 'sdxl', regions: [region()] });
    const posCond = g.nodes.rg_pos_cond_r1;
    expect(posCond.type).toBe('sdxl_compel_prompt');
    expect(posCond.prompt).toBe('a cat');
    expect(posCond.style).toBe('a cat');
    expect(hasEdge(g, 'clip_skip', 'clip', 'rg_pos_cond_r1', 'clip')).toBe(true);
    expect(hasEdge(g, 'model_loader', 'clip2', 'rg_pos_cond_r1', 'clip2')).toBe(true);
  });
});

describe('addRegionalGuidance — FLUX', () => {
  it('wires a positive prompt into pos_cond_collect and copies clip/t5, no negative path', () => {
    const g = run(fluxBaseGraph(), { base: 'flux', regions: [region({ negativePrompt: 'blurry' })] });
    expect(g.nodes.rg_pos_cond_r1.type).toBe('flux_text_encoder');
    expect(hasEdge(g, 'rg_pos_cond_r1', 'conditioning', 'pos_cond_collect', 'item')).toBe(true);
    expect(hasEdge(g, 'model_loader', 't5_encoder', 'rg_pos_cond_r1', 't5_encoder')).toBe(true);
    // No regional negative conditioning on FLUX.
    expect(g.nodes.rg_neg_cond_r1).toBeUndefined();
  });

  it('ignores regional ip_adapter on FLUX and does not wire denoise.ip_adapter', () => {
    const g = run(fluxBaseGraph(), {
      base: 'flux',
      regions: [
        region({
          referenceImages: [
            {
              beginEndStepPct: [0, 1],
              clipVisionModel: 'ViT-H',
              id: 'ref1',
              imageName: 'ref.png',
              method: 'full',
              model: ipModel('flux'),
              type: 'ip_adapter',
              weight: 1,
            },
          ],
        }),
      ],
    });
    expect(g.nodes.ip_adapter_ref1).toBeUndefined();
  });

  it('wires a flux_redux reference image mask-scoped into denoise.redux_conditioning', () => {
    const g = run(fluxBaseGraph(), {
      base: 'flux',
      regions: [
        region({
          referenceImages: [
            {
              id: 'ref2',
              imageName: 'redux.png',
              model: { base: 'flux', key: 'redux', name: 'Redux', type: 'flux_redux' },
              settings: { downsampling_factor: 1, weight: 1 },
              type: 'flux_redux',
            },
          ],
        }),
      ],
    });
    const redux = g.nodes.flux_redux_ref2;
    expect(redux.type).toBe('flux_redux');
    expect(hasEdge(g, 'rg_mask_to_tensor_r1', 'mask', 'flux_redux_ref2', 'mask')).toBe(true);
    const reduxEdge = g.edges.find(
      (e) => e.destination.node_id === 'denoise_latents' && e.destination.field === 'redux_conditioning'
    );
    expect(reduxEdge).toBeDefined();
    expect(hasEdge(g, 'flux_redux_ref2', 'redux_cond', reduxEdge!.source.node_id, 'item')).toBe(true);
  });
});

describe('addRegionalGuidance — multiple regions coexist', () => {
  it('mints distinct deterministic node ids per region', () => {
    const g = run(sdBaseGraph(), {
      base: 'sd-1',
      regions: [region({ id: 'a', positivePrompt: 'cat' }), region({ id: 'b', positivePrompt: 'dog' })],
    });
    expect(g.nodes.rg_pos_cond_a.prompt).toBe('cat');
    expect(g.nodes.rg_pos_cond_b.prompt).toBe('dog');
    expect(nodesOfType(g, 'alpha_mask_to_tensor')).toHaveLength(2);
  });
});

describe('getRegionalGuidanceRejectionReason', () => {
  const params = {
    autoNegative: false,
    hasContent: true,
    layerName: 'Region 1',
    mainBase: 'sd-1',
    negativePrompt: null as string | null,
    positivePrompt: 'a cat' as string | null,
    referenceImageCount: 0,
  };

  it('accepts a valid SD region', () => {
    expect(getRegionalGuidanceRejectionReason(params)).toBeNull();
  });

  it('rejects an unsupported base', () => {
    expect(getRegionalGuidanceRejectionReason({ ...params, mainBase: 'sd-3' })).toMatch(/not supported/);
  });

  it('rejects an empty mask', () => {
    expect(getRegionalGuidanceRejectionReason({ ...params, hasContent: false })).toMatch(/no masked region/);
  });

  it('rejects a region with no prompt and no reference images', () => {
    expect(getRegionalGuidanceRejectionReason({ ...params, positivePrompt: null, referenceImageCount: 0 })).toMatch(
      /no prompt or reference/
    );
  });

  it('rejects a FLUX negative prompt / autoNegative', () => {
    expect(getRegionalGuidanceRejectionReason({ ...params, mainBase: 'flux', negativePrompt: 'blurry' })).toMatch(
      /negative prompts are not supported for FLUX/
    );
    expect(getRegionalGuidanceRejectionReason({ ...params, autoNegative: true, mainBase: 'flux' })).toMatch(
      /auto-negative is not supported for FLUX/
    );
  });
});
