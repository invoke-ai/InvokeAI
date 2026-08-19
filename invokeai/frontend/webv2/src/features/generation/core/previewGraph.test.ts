import { describe, expect, it } from 'vitest';

import type { GenerationModelCatalogItem } from './contracts';
import type { GenerateLora, GenerateWidgetValues, LoraModelConfig, MainModelConfig } from './types';

import { getDefaultGenerateSettings } from './baseGenerationPolicies';
import { compileGeneratePreviewGraph, stabilizeBackendGraphIds } from './previewGraph';

const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl-model', name: 'SDXL', type: 'main' };

const loraModelA: LoraModelConfig = { base: 'sdxl', key: 'lora-a', name: 'Lora A', type: 'lora' };
const loraModelB: LoraModelConfig = { base: 'sdxl', key: 'lora-b', name: 'Lora B', type: 'lora' };

const loraA: GenerateLora = { isEnabled: true, model: loraModelA, weight: 0.75 };
const loraB: GenerateLora = { isEnabled: true, model: loraModelB, weight: 0.5 };

const models: GenerationModelCatalogItem[] = [sdxlModel, loraModelA, loraModelB];

const createGenerateValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues => ({
  ...getDefaultGenerateSettings(sdxlModel),
  model: sdxlModel,
  modelKey: sdxlModel.key,
  positivePrompt: 'a preview prompt',
  ...overrides,
});

const storedValues = createGenerateValues({
  cfgScale: 4,
  height: 1024,
  loras: [loraA, loraB],
  seed: 123,
  shouldRandomizeSeed: true,
  steps: 28,
  width: 1024,
});

describe('compileGeneratePreviewGraph', () => {
  it('compiles a ready graph with deterministic node ids across repeated compiles', () => {
    const input = { destination: 'gallery' as const, models, storedValues, useCpuNoise: false };
    const first = compileGeneratePreviewGraph(input);
    const second = compileGeneratePreviewGraph(input);

    expect(first.status).toBe('ready');

    if (first.status !== 'ready' || second.status !== 'ready') {
      return;
    }

    expect(first.graph.id).toBe('generate-preview');
    expect(first.graph.nodes.map((node) => node.id)).toEqual(second.graph.nodes.map((node) => node.id));

    const ids = first.graph.nodes.map((node) => node.id);

    expect(ids).toContain('core_metadata');
    expect(ids).toContain('lora_selector');
    expect(ids).toContain('lora_selector_2');
    expect(ids).toContain('denoise_latents');
  });

  it('keeps the literal seed value instead of resolving a random one', () => {
    const result = compileGeneratePreviewGraph({ destination: 'gallery', models, storedValues, useCpuNoise: false });

    if (result.status !== 'ready') {
      throw new Error('expected ready');
    }

    const seedNode = result.graph.nodes.find((node) => node.id === 'seed');

    expect(seedNode?.inputs.value).toBe(123);
  });

  it('rewrites edges consistently with rewritten node ids', () => {
    const result = compileGeneratePreviewGraph({ destination: 'gallery', models, storedValues, useCpuNoise: false });

    if (result.status !== 'ready') {
      throw new Error('expected ready');
    }

    const nodeIds = new Set(result.graph.nodes.map((node) => node.id));

    for (const edge of result.graph.edges) {
      expect(nodeIds.has(edge.sourceNodeId)).toBe(true);
      expect(nodeIds.has(edge.targetNodeId)).toBe(true);
    }
  });

  it('returns invalid with reasons instead of throwing on bad settings', () => {
    // 999 is not a multiple of the SDXL dimension grid (8).
    const badValues = { ...storedValues, height: 999 };
    const result = compileGeneratePreviewGraph({
      destination: 'gallery',
      models,
      storedValues: badValues,
      useCpuNoise: false,
    });

    expect(result.status).toBe('invalid');

    if (result.status === 'invalid') {
      expect(result.reasons.length).toBeGreaterThan(0);
    }
  });

  it('returns invalid when no supported model resolves', () => {
    const result = compileGeneratePreviewGraph({
      destination: 'gallery',
      models: [],
      storedValues,
      useCpuNoise: false,
    });

    expect(result.status).toBe('invalid');
  });

  it('pins a deterministic seed across repeated compiles when generate settings are uninitialized', () => {
    const input = { destination: 'gallery' as const, models, storedValues: undefined, useCpuNoise: false };
    const first = compileGeneratePreviewGraph(input);
    const second = compileGeneratePreviewGraph(input);

    expect(first.status).toBe('ready');

    if (first.status !== 'ready' || second.status !== 'ready') {
      return;
    }

    expect(first.graph.nodes.map((node) => node.id)).toEqual(second.graph.nodes.map((node) => node.id));

    const firstSeed = first.graph.nodes.find((node) => node.id === 'seed');
    const secondSeed = second.graph.nodes.find((node) => node.id === 'seed');

    expect(firstSeed?.inputs.value).toBe(secondSeed?.inputs.value);
  });
});

describe('stabilizeBackendGraphIds', () => {
  it('rewrites createId-suffixed ids to stable prefixes in insertion order, including edges', () => {
    const graph = {
      edges: [
        {
          destination: { field: 'value', node_id: 'lora_selector_lqx1abc_a1b2c3' },
          source: { field: 'value', node_id: 'seed' },
        },
      ],
      id: 'sdxl_graph_lqx1abc_z9z9z9',
      nodes: {
        core_metadata_lqx1abc_d4e5f6: { id: 'core_metadata_lqx1abc_d4e5f6', type: 'core_metadata' },
        lora_selector_lqx1abc_a1b2c3: { id: 'lora_selector_lqx1abc_a1b2c3', type: 'lora_selector' },
        lora_selector_lqx1abc_z9y8x7: { id: 'lora_selector_lqx1abc_z9y8x7', type: 'lora_selector' },
        seed: { id: 'seed', type: 'integer' },
      },
    };

    const result = stabilizeBackendGraphIds(graph);

    expect(Object.keys(result.nodes)).toEqual(['core_metadata', 'lora_selector', 'lora_selector_2', 'seed']);
    expect(result.edges[0]).toEqual({
      destination: { field: 'value', node_id: 'lora_selector' },
      source: { field: 'value', node_id: 'seed' },
    });
  });
});
