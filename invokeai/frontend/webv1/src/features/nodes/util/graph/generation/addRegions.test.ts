import { describe, expect, it, vi } from 'vitest';

vi.mock('features/controlLayers/store/validators', () => ({
  getRegionalGuidanceWarnings: vi.fn(() => []),
}));

import { getRegionalGuidanceState } from 'features/controlLayers/store/util';
import { addRegions } from 'features/nodes/util/graph/generation/addRegions';
import { Graph } from 'features/nodes/util/graph/generation/Graph';

describe('addRegions - Krea-2', () => {
  it('adds a masked Krea-2 encoder that shares the global encoder model', async () => {
    const g = new Graph('graph');
    const modelLoader = g.addNode({
      type: 'krea2_model_loader',
      id: 'model-loader',
      model: {
        key: 'krea',
        hash: 'hash',
        name: 'Krea-2',
        base: 'krea-2',
        type: 'main',
      },
    });
    const posCond = g.addNode({ type: 'krea2_text_encoder', id: 'global-positive', prompt: 'global' });
    const posCondCollect = g.addNode({ type: 'collect', id: 'positive-collector' });
    const ipAdapterCollect = g.addNode({ type: 'collect', id: 'ip-adapter-collector' });
    g.addEdge(modelLoader, 'qwen3_vl_encoder', posCond, 'qwen3_vl_encoder');
    g.addEdge(posCond, 'conditioning', posCondCollect, 'item');

    const manager = {
      adapters: {
        regionMasks: {
          get: vi.fn(() => ({
            renderer: {
              rasterize: vi.fn(() => Promise.resolve({ image_name: 'region-mask.png' })),
            },
          })),
        },
      },
    };

    await addRegions({
      manager: manager as never,
      regions: [getRegionalGuidanceState('region', { positivePrompt: 'red fox' })],
      g,
      bbox: { x: 0, y: 0, width: 1024, height: 1024 },
      model: { base: 'krea-2' } as never,
      posCond,
      negCond: null,
      posCondCollect,
      negCondCollect: null,
      ipAdapterCollect,
      fluxReduxCollect: null,
    });

    const graph = g.getGraph();
    const regionalEncoder = Object.values(graph.nodes).find(
      (node) => node.type === 'krea2_text_encoder' && node.id !== posCond.id
    );
    expect(regionalEncoder).toBeDefined();
    expect(regionalEncoder).toMatchObject({ prompt: 'red fox' });
    expect(graph.edges).toContainEqual(
      expect.objectContaining({
        source: { node_id: modelLoader.id, field: 'qwen3_vl_encoder' },
        destination: { node_id: regionalEncoder!.id, field: 'qwen3_vl_encoder' },
      })
    );
    expect(graph.edges).toContainEqual(
      expect.objectContaining({
        destination: { node_id: regionalEncoder!.id, field: 'mask' },
      })
    );
    expect(graph.edges).toContainEqual(
      expect.objectContaining({
        source: { node_id: regionalEncoder!.id, field: 'conditioning' },
        destination: { node_id: posCondCollect.id, field: 'item' },
      })
    );
  });

  it('collects the transformed regional conditioning when a transform is provided', async () => {
    const g = new Graph('graph');
    const modelLoader = g.addNode({
      type: 'krea2_model_loader',
      id: 'model-loader',
      model: {
        key: 'krea',
        hash: 'hash',
        name: 'Krea-2',
        base: 'krea-2',
        type: 'main',
      },
    });
    const posCond = g.addNode({ type: 'krea2_text_encoder', id: 'global-positive', prompt: 'global' });
    const posCondCollect = g.addNode({ type: 'collect', id: 'positive-collector' });
    const ipAdapterCollect = g.addNode({ type: 'collect', id: 'ip-adapter-collector' });
    g.addEdge(modelLoader, 'qwen3_vl_encoder', posCond, 'qwen3_vl_encoder');

    const manager = {
      adapters: {
        regionMasks: {
          get: vi.fn(() => ({
            renderer: {
              rasterize: vi.fn(() => Promise.resolve({ image_name: 'region-mask.png' })),
            },
          })),
        },
      },
    };
    const transformRegionalPositiveConditioning = vi.fn((regionalPosCond: { id: string; type: string }) => {
      const seedVariance = g.addNode({
        type: 'krea2_seed_variance',
        id: 'regional-seed-variance',
        strength: 0.5,
        randomize_percent: 50,
      });
      g.addEdgeFromObj({
        source: { node_id: regionalPosCond.id, field: 'conditioning' },
        destination: { node_id: seedVariance.id, field: 'conditioning' },
      });
      return seedVariance;
    });

    await addRegions({
      manager: manager as never,
      regions: [getRegionalGuidanceState('region', { positivePrompt: 'red fox' })],
      g,
      bbox: { x: 0, y: 0, width: 1024, height: 1024 },
      model: { base: 'krea-2' } as never,
      posCond,
      negCond: null,
      posCondCollect,
      negCondCollect: null,
      ipAdapterCollect,
      fluxReduxCollect: null,
      transformRegionalPositiveConditioning,
    });

    const graph = g.getGraph();
    const regionalEncoder = Object.values(graph.nodes).find(
      (node) => node.type === 'krea2_text_encoder' && node.id !== posCond.id
    );
    expect(transformRegionalPositiveConditioning).toHaveBeenCalledWith(regionalEncoder);
    expect(graph.edges).not.toContainEqual(
      expect.objectContaining({
        source: { node_id: regionalEncoder!.id, field: 'conditioning' },
        destination: { node_id: posCondCollect.id, field: 'item' },
      })
    );
    expect(graph.edges).toContainEqual(
      expect.objectContaining({
        source: { node_id: 'regional-seed-variance', field: 'conditioning' },
        destination: { node_id: posCondCollect.id, field: 'item' },
      })
    );
  });
});
