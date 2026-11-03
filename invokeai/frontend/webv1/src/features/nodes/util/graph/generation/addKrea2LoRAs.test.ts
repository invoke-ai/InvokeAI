import type { Invocation } from 'services/api/types';
import { describe, expect, it, vi } from 'vitest';

let nextId = 0;
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:${nextId++}`,
}));

import { addKrea2LoRAs } from './addKrea2LoRAs';
import { Graph } from './Graph';

const model = { key: 'krea2-model', hash: 'h', name: 'Krea-2', base: 'krea-2', type: 'main' };

const buildBaseGraph = (withNeg: boolean) => {
  nextId = 0;
  const g = new Graph('test');
  const modelLoader = g.addNode({
    type: 'krea2_model_loader',
    id: 'model_loader',
    model,
  } as Invocation<'krea2_model_loader'>);
  const denoise = g.addNode({ type: 'krea2_denoise', id: 'denoise' } as Invocation<'krea2_denoise'>);
  const posCond = g.addNode({ type: 'krea2_text_encoder', id: 'pos_prompt' } as Invocation<'krea2_text_encoder'>);
  const negCond = withNeg
    ? g.addNode({ type: 'krea2_text_encoder', id: 'neg_prompt' } as Invocation<'krea2_text_encoder'>)
    : null;

  // The direct model -> denoise / text-encoder edges that addKrea2LoRAs must reroute.
  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'qwen3_vl_encoder', posCond, 'qwen3_vl_encoder');
  if (negCond) {
    g.addEdge(modelLoader, 'qwen3_vl_encoder', negCond, 'qwen3_vl_encoder');
  }

  return { g, modelLoader, denoise, posCond, negCond };
};

const stateWith = (loras: unknown[]) => ({ loras: { loras } }) as never;

const enabledKrea2Lora = {
  id: 'lora-1',
  isEnabled: true,
  weight: 0.8,
  model: { key: 'lora-key', hash: 'lora-hash', name: 'My Krea LoRA', base: 'krea-2', type: 'lora' },
};

describe('addKrea2LoRAs', () => {
  it('reroutes the transformer and both text encoders through the collection loader', () => {
    const { g, modelLoader, denoise, posCond, negCond } = buildBaseGraph(true);

    addKrea2LoRAs(stateWith([enabledKrea2Lora]), g, denoise, modelLoader, posCond, negCond);

    const graph = g.getGraph();
    const nodeTypes = Object.values(graph.nodes).map((n) => n.type);
    expect(nodeTypes).toContain('krea2_lora_collection_loader');
    expect(nodeTypes).toContain('collect');
    expect(nodeTypes).toContain('lora_selector');

    const collectionLoaderId = Object.values(graph.nodes).find((n) => n.type === 'krea2_lora_collection_loader')!.id;

    // The single edge into denoise.transformer now originates from the collection loader (old direct edge gone).
    const transformerEdges = graph.edges.filter(
      (e) => e.destination.node_id === denoise.id && e.destination.field === 'transformer'
    );
    expect(transformerEdges).toHaveLength(1);
    expect(transformerEdges[0]!.source.node_id).toBe(collectionLoaderId);

    // Both the positive and negative encoders are rerouted to the collection loader.
    for (const cond of [posCond, negCond!]) {
      const encoderEdges = graph.edges.filter(
        (e) => e.destination.node_id === cond.id && e.destination.field === 'qwen3_vl_encoder'
      );
      expect(encoderEdges).toHaveLength(1);
      expect(encoderEdges[0]!.source.node_id).toBe(collectionLoaderId);
    }

    // No stale direct edge remains from the model loader to denoise or the encoders.
    const staleDirectEdges = graph.edges.filter(
      (e) =>
        e.source.node_id === modelLoader.id && [denoise.id, posCond.id, negCond!.id].includes(e.destination.node_id)
    );
    expect(staleDirectEdges).toHaveLength(0);

    // The lora selector feeds the collector, which feeds the collection loader.
    const selectorId = Object.values(graph.nodes).find((n) => n.type === 'lora_selector')!.id;
    const collectorId = Object.values(graph.nodes).find((n) => n.type === 'collect')!.id;
    expect(graph.edges.some((e) => e.source.node_id === selectorId && e.destination.node_id === collectorId)).toBe(
      true
    );
    expect(
      graph.edges.some((e) => e.source.node_id === collectorId && e.destination.node_id === collectionLoaderId)
    ).toBe(true);
  });

  it('does nothing (keeps direct edges) when no Krea-2 LoRAs are enabled', () => {
    const { g, modelLoader, denoise, posCond, negCond } = buildBaseGraph(true);

    addKrea2LoRAs(stateWith([]), g, denoise, modelLoader, posCond, negCond);

    const graph = g.getGraph();
    expect(Object.values(graph.nodes).map((n) => n.type)).not.toContain('krea2_lora_collection_loader');
    const transformerEdges = graph.edges.filter(
      (e) => e.destination.node_id === denoise.id && e.destination.field === 'transformer'
    );
    expect(transformerEdges).toHaveLength(1);
    expect(transformerEdges[0]!.source.node_id).toBe(modelLoader.id);
  });

  it('ignores disabled and non-Krea-2 LoRAs', () => {
    const { g, modelLoader, denoise, posCond, negCond } = buildBaseGraph(false);

    addKrea2LoRAs(
      stateWith([
        { ...enabledKrea2Lora, isEnabled: false },
        { ...enabledKrea2Lora, id: 'flux', model: { ...enabledKrea2Lora.model, base: 'flux' } },
      ]),
      g,
      denoise,
      modelLoader,
      posCond,
      negCond
    );

    expect(Object.values(g.getGraph().nodes).map((n) => n.type)).not.toContain('krea2_lora_collection_loader');
  });
});
