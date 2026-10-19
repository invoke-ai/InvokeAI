import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import type { AddControlLayersOptions, ControlLayerGraphInput, ControlModelIdentifier } from './addControlLayers';

import { addControlLayers, CONTROL_DENOISE_NODE_ID, isControlKindSupportedForBase } from './addControlLayers';

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

const model = (base: string, type = 'controlnet'): ControlModelIdentifier => ({
  key: `model-${base}`,
  name: `Model ${base}`,
  base,
  type,
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

seedArchitectureCapabilities();

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

  it('anima_lllite is supported on anima only, which supports nothing else', () => {
    expect(isControlKindSupportedForBase('anima', 'anima_lllite')).toBe(true);
    expect(isControlKindSupportedForBase('sdxl', 'anima_lllite')).toBe(false);
    expect(isControlKindSupportedForBase('anima', 'controlnet')).toBe(false);
  });

  it('control_lora is supported on flux only', () => {
    expect(isControlKindSupportedForBase('flux', 'control_lora')).toBe(true);
    expect(isControlKindSupportedForBase('sd-1', 'control_lora')).toBe(false);
    expect(isControlKindSupportedForBase('sdxl', 'control_lora')).toBe(false);
    expect(isControlKindSupportedForBase('sd-2', 'control_lora')).toBe(false);
  });
});

describe('CONTROL_DENOISE_NODE_ID', () => {
  it('is the deterministic denoise node id', () => {
    expect(CONTROL_DENOISE_NODE_ID).toBe('denoise_latents');
  });
});

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

describe('addControlLayers — t2i_adapter on sdxl', () => {
  it('creates a t2i_adapter node with exact fields and wires collector to denoise.t2i_adapter', () => {
    const m = model('sdxl', 't2i_adapter');
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

describe('addControlLayers — control_lora on flux', () => {
  it('creates a flux_control_lora_loader wired directly to denoise.control_lora with no collector', () => {
    const m = model('flux', 'control_lora');
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

describe('addControlLayers — control_lora limits', () => {
  it('rejects a second control_lora with the shared reason code', () => {
    expect(() =>
      run({
        base: 'flux',
        layers: [
          layer({ id: 'CL1', kind: 'control_lora', model: model('flux', 'control_lora') }),
          layer({ id: 'CL2', kind: 'control_lora', model: model('flux', 'control_lora') }),
        ],
      })
    ).toThrow(/control_lora_limit/);
  });

  it('rejects control_lora for a dev_fill main model variant', () => {
    expect(() =>
      run({
        base: 'flux',
        modelVariant: 'dev_fill',
        layers: [layer({ id: 'CL1', kind: 'control_lora', model: model('flux', 'control_lora') })],
      })
    ).toThrow(/flux_fill_control_lora/);
  });
});

describe('addControlLayers — Z-Image control', () => {
  it('creates the exact backend node and connects it directly to Z-Image denoise', () => {
    const m = model('z-image', 'controlnet');
    const graph = run({
      base: 'z-image',
      layers: [
        layer({
          beginEndStepPct: [0.2, 0.9],
          controlMode: null,
          id: 'Z1',
          imageName: 'z-control.png',
          kind: 'z_image_control',
          model: m,
          weight: 0.7,
        }),
      ],
    });

    expect(graph.nodes.z_image_control_Z1).toEqual({
      begin_step_percent: 0.2,
      control_context_scale: 0.7,
      control_model: m,
      end_step_percent: 0.9,
      id: 'z_image_control_Z1',
      image: { image_name: 'z-control.png' },
      is_intermediate: true,
      type: 'z_image_control',
      use_cache: true,
    });
    expect(graph.edges).toContainEqual({
      destination: { field: 'control', node_id: 'denoise_latents' },
      source: { field: 'control', node_id: 'z_image_control_Z1' },
    });
    expect(Object.values(graph.nodes).some((node) => node.type === 'collect')).toBe(false);
  });

  it('rejects a second Z-Image control because denoise accepts a single field', () => {
    const m = model('z-image', 'controlnet');
    expect(() =>
      run({
        base: 'z-image',
        layers: [
          layer({ id: 'Z1', kind: 'z_image_control', model: m }),
          layer({ id: 'Z2', kind: 'z_image_control', model: m }),
        ],
      })
    ).toThrow(/z_image_control_limit/);
  });
});

describe('addControlLayers — Anima ControlNet-LLLite', () => {
  const lllite = (key: string): ControlModelIdentifier => ({
    base: 'anima',
    hash: `hash-${key}`,
    key,
    name: `LLLite ${key}`,
    type: 'controlnet',
  });
  const lliteLayer = (id: string, key: string, overrides: Partial<ControlLayerGraphInput> = {}) =>
    layer({
      beginEndStepPct: [0, 1],
      controlMode: null,
      id,
      imageName: `${id}.png`,
      kind: 'anima_lllite',
      model: lllite(key),
      modelCondInChannels: 3,
      weight: 1,
      ...overrides,
    });
  const runAnima = (layers: ControlLayerGraphInput[]) => run({ base: 'anima', layers });

  it('adds nothing for an Anima graph without control layers', () => {
    expect(runAnima([])).toEqual(baseGraph());
  });

  it('builds one anima_lllite node per layer with the backend field names, without a mask', () => {
    const graph = runAnima([lliteLayer('A', 'sketch', { beginEndStepPct: [0.1, 0.8], weight: 0.6 })]);

    expect(graph.nodes.anima_lllite_A).toEqual({
      begin_step_percent: 0.1,
      control_model: lllite('sketch'),
      end_step_percent: 0.8,
      id: 'anima_lllite_A',
      image: { image_name: 'A.png' },
      is_intermediate: true,
      type: 'anima_lllite',
      use_cache: true,
      weight: 0.6,
    });
    expect(graph.nodes.control_lllite_collector).toEqual({
      id: 'control_lllite_collector',
      is_intermediate: true,
      type: 'collect',
      use_cache: true,
    });
    expect(graph.edges).toEqual([
      {
        destination: { field: 'control_lllite', node_id: 'denoise_latents' },
        source: { field: 'collection', node_id: 'control_lllite_collector' },
      },
      {
        destination: { field: 'item', node_id: 'control_lllite_collector' },
        source: { field: 'control', node_id: 'anima_lllite_A' },
      },
    ]);
  });

  it('fans several adapters into one collector feeding denoise.control_lllite once', () => {
    const graph = runAnima([
      lliteLayer('A', 'sketch'),
      lliteLayer('B', 'depth', { weight: -1 }),
      lliteLayer('C', 'pose', { weight: 2 }),
    ]);

    expect(Object.values(graph.nodes).filter((node) => node.type === 'anima_lllite')).toHaveLength(3);
    expect(Object.values(graph.nodes).filter((node) => node.type === 'collect')).toHaveLength(1);
    expect(edgesTo(graph, 'control_lllite_collector', 'item').map((edge) => edge.source)).toEqual([
      { field: 'control', node_id: 'anima_lllite_A' },
      { field: 'control', node_id: 'anima_lllite_B' },
      { field: 'control', node_id: 'anima_lllite_C' },
    ]);
    expect(edgesTo(graph, 'denoise_latents', 'control_lllite')).toHaveLength(1);
    expect(graph.nodes.anima_lllite_B?.weight).toBe(-1);
    expect(graph.nodes.anima_lllite_C?.weight).toBe(2);
    // ControlNet-only fields never reach the LLLite node.
    expect(graph.nodes.anima_lllite_A).not.toHaveProperty('control_mode');
    expect(graph.nodes.anima_lllite_A).not.toHaveProperty('mask');
  });

  it('rejects a model the denoiser would apply twice', () => {
    expect(() => runAnima([lliteLayer('A', 'sketch'), lliteLayer('B', 'sketch')])).toThrow(/duplicate_lllite_model/);
  });

  it.each([
    [{ modelCondInChannels: 4 }, /lllite_inpaint_adapter/],
    [{ modelCondInChannels: null }, /lllite_channels_unknown/],
    [{ modelCondInChannels: undefined }, /lllite_channels_unknown/],
    [{ weight: 2.01 }, /invalid_adapter_values/],
    [{ weight: -1.01 }, /invalid_adapter_values/],
    [{ beginEndStepPct: [0.5, 0.5] as [number, number] }, /invalid_adapter_values/],
    [{ model: { ...lllite('sd'), base: 'sd-1' } }, /incompatible_base/],
    [{ kind: 'controlnet' as const }, /switch_adapter_kind/],
  ])('rejects %o', (overrides, reason) => {
    expect(() => runAnima([lliteLayer('A', 'sketch', overrides)])).toThrow(reason);
  });

  it('rejects an LLLite layer on another base', () => {
    expect(() => run({ base: 'sdxl', layers: [lliteLayer('A', 'sketch', { model: lllite('x') })] })).toThrow(
      /switch_adapter_kind/
    );
  });
});

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

describe('addControlLayers — unsupported kind rejected', () => {
  it('rejects a t2i_adapter layer on flux, which runs another kind', () => {
    expect(() =>
      run({
        base: 'flux',
        layers: [layer({ id: 'X', kind: 't2i_adapter', model: model('flux', 't2i_adapter') })],
      })
    ).toThrow(/switch_adapter_kind/);
  });

  it('rejects a mixed set containing an unsupported layer', () => {
    expect(() =>
      run({
        base: 'flux',
        layers: [
          layer({ id: 'skip', kind: 't2i_adapter', model: model('flux', 't2i_adapter') }),
          layer({ id: 'keep', kind: 'controlnet', model: model('flux') }),
        ],
      })
    ).toThrow(/switch_adapter_kind/);
  });

  it('rejects a graph input whose resolved model has an incompatible base', () => {
    expect(() => run({ base: 'sd-1', layers: [layer({ model: model('sdxl') })] })).toThrow(/incompatible_base/);
  });

  it('rejects malformed numeric adapter values at the graph boundary', () => {
    expect(() => run({ base: 'sd-1', layers: [layer({ weight: Number.NaN })] })).toThrow(/invalid_adapter_values/);
  });
});

describe('addControlLayers — missing denoise node', () => {
  it('throws when the base graph has no denoise node', () => {
    const graph: TestGraph = { id: 'g', nodes: {}, edges: [] };
    expect(() => addControlLayers(graph as never, { base: 'sd-1', layers: [layer()] })).toThrow(
      /missing the denoise node/
    );
  });
});
