import { describe, expect, it } from 'vitest';

import type {
  AddControlLayersOptions,
  ControlAdapterKind,
  ControlLayerGraphInput,
  ControlModelIdentifier,
} from './addControlLayers';

import {
  addControlLayers,
  CONTROL_DENOISE_NODE_ID,
  getControlLayerRejectionReason,
  isControlKindSupportedForBase,
} from './addControlLayers';

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

interface TestGraph {
  id: string;
  nodes: Record<string, { id: string; type: string; [key: string]: unknown }>;
  edges: {
    source: { node_id: string; field: string };
    destination: { node_id: string; field: string };
  }[];
}

/** A minimal built base graph containing only the required denoise node. */
const baseGraph = (): TestGraph => ({
  id: 'g',
  nodes: {
    denoise_latents: { id: 'denoise_latents', type: 'denoise_latents' },
  },
  edges: [],
});

const model = (base: string): ControlModelIdentifier => ({
  key: `model-${base}`,
  name: `Model ${base}`,
  base,
  type: 'controlnet',
});

const layer = (overrides: Partial<ControlLayerGraphInput> = {}): ControlLayerGraphInput => ({
  id: 'layer1',
  imageName: 'image1.png',
  kind: 'controlnet',
  model: model('sd-1'),
  weight: 0.75,
  beginEndStepPct: [0.1, 0.8],
  controlMode: 'balanced',
  ...overrides,
});

// Convenience to run addControlLayers against a fresh graph.
const run = (options: AddControlLayersOptions) => {
  const graph = baseGraph();
  // Cast: TestGraph is structurally the BackendGraphContract shape addControlLayers needs.
  addControlLayers(graph as never, options);
  return graph;
};

const edgesTo = (graph: TestGraph, nodeId: string, field: string) =>
  graph.edges.filter((e) => e.destination.node_id === nodeId && e.destination.field === field);

// ---------------------------------------------------------------------------
// 1. isControlKindSupportedForBase full matrix
// ---------------------------------------------------------------------------

describe('isControlKindSupportedForBase', () => {
  it('controlnet is supported on sd-1, sdxl, flux only', () => {
    expect(isControlKindSupportedForBase('sd-1', 'controlnet')).toBe(true);
    expect(isControlKindSupportedForBase('sdxl', 'controlnet')).toBe(true);
    expect(isControlKindSupportedForBase('flux', 'controlnet')).toBe(true);
    expect(isControlKindSupportedForBase('sd-2', 'controlnet')).toBe(false);
    expect(isControlKindSupportedForBase('sd-3', 'controlnet')).toBe(false);
  });

  it('t2i_adapter is supported on sd-1 and sdxl only (not flux)', () => {
    expect(isControlKindSupportedForBase('sd-1', 't2i_adapter')).toBe(true);
    expect(isControlKindSupportedForBase('sdxl', 't2i_adapter')).toBe(true);
    expect(isControlKindSupportedForBase('flux', 't2i_adapter')).toBe(false);
    expect(isControlKindSupportedForBase('sd-2', 't2i_adapter')).toBe(false);
  });

  it('control_lora is supported on flux only', () => {
    expect(isControlKindSupportedForBase('flux', 'control_lora')).toBe(true);
    expect(isControlKindSupportedForBase('sd-1', 'control_lora')).toBe(false);
    expect(isControlKindSupportedForBase('sdxl', 'control_lora')).toBe(false);
    expect(isControlKindSupportedForBase('sd-2', 'control_lora')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// CONTROL_DENOISE_NODE_ID sanity
// ---------------------------------------------------------------------------

describe('CONTROL_DENOISE_NODE_ID', () => {
  it('is the deterministic denoise node id', () => {
    expect(CONTROL_DENOISE_NODE_ID).toBe('denoise_latents');
  });
});

// ---------------------------------------------------------------------------
// 2. controlnet on sd-1
// ---------------------------------------------------------------------------

describe('addControlLayers — controlnet on sd-1', () => {
  it('creates a controlnet node with exact fields and wires the collector to denoise.control', () => {
    const m = model('sd-1');
    const graph = run({
      base: 'sd-1',
      layers: [
        layer({
          id: 'L1',
          imageName: 'ctrl.png',
          model: m,
          weight: 0.6,
          beginEndStepPct: [0.2, 0.9],
          controlMode: null,
        }),
      ],
    });

    const node = graph.nodes['control_net_L1'];
    expect(node).toBeDefined();
    expect(node.type).toBe('controlnet');
    expect(node.begin_step_percent).toBe(0.2);
    expect(node.end_step_percent).toBe(0.9);
    expect(node.control_model).toBe(m);
    expect(node.control_weight).toBe(0.6);
    expect(node.resize_mode).toBe('just_resize');
    // controlMode null → defaults to 'balanced'
    expect(node.control_mode).toBe('balanced');
    expect(node.image).toEqual({ image_name: 'ctrl.png' });

    // Collector created and wired.
    expect(graph.nodes['control_net_collector']).toBeDefined();
    expect(graph.nodes['control_net_collector'].type).toBe('collect');

    // node.control → collector.item
    expect(
      graph.edges.some(
        (e) =>
          e.source.node_id === 'control_net_L1' &&
          e.source.field === 'control' &&
          e.destination.node_id === 'control_net_collector' &&
          e.destination.field === 'item'
      )
    ).toBe(true);

    // collector.collection → denoise.control
    expect(
      graph.edges.some(
        (e) =>
          e.source.node_id === 'control_net_collector' &&
          e.source.field === 'collection' &&
          e.destination.node_id === 'denoise_latents' &&
          e.destination.field === 'control'
      )
    ).toBe(true);
  });

  it('carries an explicit controlMode when provided', () => {
    const graph = run({
      base: 'sd-1',
      layers: [layer({ id: 'L1', controlMode: 'more_control' })],
    });
    expect(graph.nodes['control_net_L1'].control_mode).toBe('more_control');
  });
});

// ---------------------------------------------------------------------------
// 3. controlnet on flux
// ---------------------------------------------------------------------------

describe('addControlLayers — controlnet on flux', () => {
  it('uses flux_controlnet type and omits control_mode entirely', () => {
    const m = model('flux');
    const graph = run({
      base: 'flux',
      layers: [layer({ id: 'F1', kind: 'controlnet', model: m, controlMode: 'more_prompt' })],
    });

    const node = graph.nodes['control_net_F1'];
    expect(node).toBeDefined();
    expect(node.type).toBe('flux_controlnet');
    expect(node.resize_mode).toBe('just_resize');
    expect(node.control_model).toBe(m);
    // No control_mode key at all, even though controlMode was set.
    expect(node).not.toHaveProperty('control_mode');

    // Collector wired same as SD family.
    expect(
      graph.edges.some(
        (e) =>
          e.source.node_id === 'control_net_collector' &&
          e.source.field === 'collection' &&
          e.destination.node_id === 'denoise_latents' &&
          e.destination.field === 'control'
      )
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 4. t2i_adapter on sdxl
// ---------------------------------------------------------------------------

describe('addControlLayers — t2i_adapter on sdxl', () => {
  it('creates a t2i_adapter node with exact fields and wires collector to denoise.t2i_adapter', () => {
    const m = model('sdxl');
    const graph = run({
      base: 'sdxl',
      layers: [
        layer({
          id: 'T1',
          kind: 't2i_adapter',
          imageName: 't2i.png',
          model: m,
          weight: 0.5,
          beginEndStepPct: [0.05, 0.7],
        }),
      ],
    });

    const node = graph.nodes['t2i_adapter_T1'];
    expect(node).toBeDefined();
    expect(node.type).toBe('t2i_adapter');
    expect(node.begin_step_percent).toBe(0.05);
    expect(node.end_step_percent).toBe(0.7);
    expect(node.resize_mode).toBe('just_resize');
    expect(node.t2i_adapter_model).toBe(m);
    expect(node.weight).toBe(0.5);
    expect(node.image).toEqual({ image_name: 't2i.png' });
    // t2i_adapter has no control_mode / control_model / control_weight
    expect(node).not.toHaveProperty('control_mode');
    expect(node).not.toHaveProperty('control_model');

    expect(graph.nodes['t2i_adapter_collector']).toBeDefined();
    expect(graph.nodes['t2i_adapter_collector'].type).toBe('collect');

    // node.t2i_adapter → collector.item
    expect(
      graph.edges.some(
        (e) =>
          e.source.node_id === 't2i_adapter_T1' &&
          e.source.field === 't2i_adapter' &&
          e.destination.node_id === 't2i_adapter_collector' &&
          e.destination.field === 'item'
      )
    ).toBe(true);

    // collector.collection → denoise.t2i_adapter
    expect(
      graph.edges.some(
        (e) =>
          e.source.node_id === 't2i_adapter_collector' &&
          e.source.field === 'collection' &&
          e.destination.node_id === 'denoise_latents' &&
          e.destination.field === 't2i_adapter'
      )
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 5. control_lora on flux
// ---------------------------------------------------------------------------

describe('addControlLayers — control_lora on flux', () => {
  it('creates a flux_control_lora_loader wired directly to denoise.control_lora with no collector', () => {
    const m = model('flux');
    const graph = run({
      base: 'flux',
      layers: [layer({ id: 'CL1', kind: 'control_lora', imageName: 'lora.png', model: m, weight: 0.9 })],
    });

    const node = graph.nodes['control_lora_CL1'];
    expect(node).toBeDefined();
    expect(node.type).toBe('flux_control_lora_loader');
    expect(node.lora).toBe(m);
    expect(node.image).toEqual({ image_name: 'lora.png' });
    expect(node.weight).toBe(0.9);

    // Wired directly node.control_lora → denoise.control_lora
    expect(
      graph.edges.some(
        (e) =>
          e.source.node_id === 'control_lora_CL1' &&
          e.source.field === 'control_lora' &&
          e.destination.node_id === 'denoise_latents' &&
          e.destination.field === 'control_lora'
      )
    ).toBe(true);

    // No collector node of any kind created.
    expect(graph.nodes['control_net_collector']).toBeUndefined();
    expect(graph.nodes['t2i_adapter_collector']).toBeUndefined();
    expect(Object.values(graph.nodes).some((n) => n.type === 'collect')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// 6. control_lora limits: second skipped; dev_fill skips
// ---------------------------------------------------------------------------

describe('addControlLayers — control_lora limits', () => {
  it('adds at most one control_lora; the second layer is skipped', () => {
    const graph = run({
      base: 'flux',
      layers: [
        layer({ id: 'CL1', kind: 'control_lora', model: model('flux') }),
        layer({ id: 'CL2', kind: 'control_lora', model: model('flux') }),
      ],
    });

    const loraNodes = Object.values(graph.nodes).filter((n) => n.type === 'flux_control_lora_loader');
    expect(loraNodes).toHaveLength(1);
    expect(graph.nodes['control_lora_CL1']).toBeDefined();
    expect(graph.nodes['control_lora_CL2']).toBeUndefined();
  });

  it('skips control_lora entirely for a dev_fill main model variant', () => {
    const graph = run({
      base: 'flux',
      modelVariant: 'dev_fill',
      layers: [layer({ id: 'CL1', kind: 'control_lora', model: model('flux') })],
    });

    expect(Object.values(graph.nodes).some((n) => n.type === 'flux_control_lora_loader')).toBe(false);
    // Only the denoise node remains.
    expect(Object.keys(graph.nodes)).toEqual(['denoise_latents']);
    expect(graph.edges).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// 7. Two controlnet layers → distinct nodes, one shared collector
// ---------------------------------------------------------------------------

describe('addControlLayers — per-layer separation', () => {
  it('creates two distinct adapter nodes feeding one shared collector', () => {
    const graph = run({
      base: 'sd-1',
      layers: [layer({ id: 'A', imageName: 'a.png' }), layer({ id: 'B', imageName: 'b.png' })],
    });

    expect(graph.nodes['control_net_A']).toBeDefined();
    expect(graph.nodes['control_net_B']).toBeDefined();
    expect(graph.nodes['control_net_A'].image).toEqual({ image_name: 'a.png' });
    expect(graph.nodes['control_net_B'].image).toEqual({ image_name: 'b.png' });

    // Exactly one collector.
    const collectors = Object.values(graph.nodes).filter((n) => n.type === 'collect');
    expect(collectors).toHaveLength(1);

    // Two item edges into the shared collector.
    const itemEdges = graph.edges.filter(
      (e) => e.destination.node_id === 'control_net_collector' && e.destination.field === 'item'
    );
    expect(itemEdges).toHaveLength(2);
    expect(itemEdges.map((e) => e.source.node_id).sort()).toEqual(['control_net_A', 'control_net_B']);

    // Only one collection→denoise edge (collector wired once).
    expect(edgesTo(graph, 'denoise_latents', 'control')).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// 8. Unsupported kind for base is silently skipped
// ---------------------------------------------------------------------------

describe('addControlLayers — unsupported kind skipped', () => {
  it('adds nothing for a t2i_adapter layer on flux', () => {
    const graph = run({
      base: 'flux',
      layers: [layer({ id: 'X', kind: 't2i_adapter', model: model('flux') })],
    });

    expect(Object.keys(graph.nodes)).toEqual(['denoise_latents']);
    expect(graph.edges).toHaveLength(0);
  });

  it('skips only unsupported layers while keeping supported ones', () => {
    const graph = run({
      base: 'flux',
      layers: [
        layer({ id: 'skip', kind: 't2i_adapter', model: model('flux') }),
        layer({ id: 'keep', kind: 'controlnet', model: model('flux') }),
      ],
    });

    expect(graph.nodes['t2i_adapter_skip']).toBeUndefined();
    expect(graph.nodes['control_net_keep']).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// 9. throws when denoise node missing
// ---------------------------------------------------------------------------

describe('addControlLayers — missing denoise node', () => {
  it('throws when the base graph has no denoise node', () => {
    const graph: TestGraph = { id: 'g', nodes: {}, edges: [] };
    expect(() => addControlLayers(graph as never, { base: 'sd-1', layers: [layer()] })).toThrow(
      /missing the denoise node/
    );
  });
});

// ---------------------------------------------------------------------------
// 10. getControlLayerRejectionReason branches
// ---------------------------------------------------------------------------

describe('getControlLayerRejectionReason', () => {
  const validParams = {
    layerName: 'My Layer',
    hasContent: true,
    kind: 'controlnet' as ControlAdapterKind,
    adapterModel: { base: 'sd-1' },
    mainBase: 'sd-1',
    mainVariant: undefined as string | undefined,
  };

  it('returns null for a valid controlnet on sd-1 with a matching model base', () => {
    expect(getControlLayerRejectionReason(validParams)).toBeNull();
  });

  it('rejects when the layer has no content', () => {
    const reason = getControlLayerRejectionReason({ ...validParams, hasContent: false });
    expect(reason).toEqual(expect.any(String));
    expect(reason).toContain('no control content');
  });

  it('rejects when no adapter model is selected', () => {
    const reason = getControlLayerRejectionReason({ ...validParams, adapterModel: null });
    expect(reason).toEqual(expect.any(String));
    expect(reason).toContain('no control model');
  });

  it('rejects when the base+kind is unsupported', () => {
    const reason = getControlLayerRejectionReason({
      ...validParams,
      mainBase: 'sd-2',
      adapterModel: { base: 'sd-2' },
    });
    expect(reason).toEqual(expect.any(String));
    expect(reason).toContain('not supported');
  });

  it('rejects when the adapter model base does not match the main base', () => {
    const reason = getControlLayerRejectionReason({
      ...validParams,
      adapterModel: { base: 'sdxl' },
    });
    expect(reason).toEqual(expect.any(String));
    expect(reason).toContain('incompatible base');
  });

  it('rejects FLUX Fill (dev_fill) + control_lora', () => {
    const reason = getControlLayerRejectionReason({
      layerName: 'My Layer',
      hasContent: true,
      kind: 'control_lora',
      adapterModel: { base: 'flux' },
      mainBase: 'flux',
      mainVariant: 'dev_fill',
    });
    expect(reason).toEqual(expect.any(String));
    expect(reason).toContain('FLUX Fill');
  });
});
