import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { NodesDataPort } from './transport';

const adapter = vi.hoisted(() => ({ request: vi.fn(), requestJson: vi.fn() }));

vi.mock('./transport', () => ({
  browserNodesDataPort: {
    buildUrl: (path: string) => path,
    request: adapter.request,
    requestJson: adapter.requestJson,
  } satisfies NodesDataPort,
}));

beforeEach(() => {
  adapter.request.mockReset().mockResolvedValue(new Response());
  adapter.requestJson.mockReset();
});

describe('custom node data adapter', () => {
  it('normalizes catalog DTOs returned by the fixture adapter', async () => {
    adapter.requestJson.mockResolvedValue({
      custom_nodes_path: '/nodes',
      node_packs: [{ name: 'pack', node_count: 2, node_types: ['one', 'two'], path: '/nodes/pack' }],
    });
    const { listCustomNodePacks } = await import('./api');

    await expect(listCustomNodePacks()).resolves.toEqual({
      customNodesPath: '/nodes',
      nodePacks: [{ name: 'pack', nodeCount: 2, nodeTypes: ['one', 'two'], path: '/nodes/pack' }],
    });
    expect(adapter.requestJson).toHaveBeenCalledWith('/api/v2/custom_nodes/');
  });

  it('encodes pack identifiers and sends install intent in the body', async () => {
    adapter.requestJson.mockResolvedValue({ success: true });
    const { installCustomNodePack, uninstallCustomNodePack } = await import('./api');

    await installCustomNodePack('https://example.test/nodes.git');
    await uninstallCustomNodePack('pack/with spaces');

    expect(adapter.requestJson).toHaveBeenNthCalledWith(1, '/api/v2/custom_nodes/install', {
      body: JSON.stringify({ source: 'https://example.test/nodes.git' }),
      method: 'POST',
    });
    expect(adapter.requestJson).toHaveBeenNthCalledWith(2, '/api/v2/custom_nodes/pack%2Fwith%20spaces', {
      method: 'DELETE',
    });
  });
});
