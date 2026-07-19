import { describe, expect, it } from 'vitest';

import { normalizeNodePackCatalog, normalizeNodePackInfo } from './catalog';

describe('node catalog normalization', () => {
  it('maps backend fields into the owned descriptor vocabulary', () => {
    expect(
      normalizeNodePackInfo({
        name: 'invokeai-node-pack',
        node_count: 3,
        node_types: ['alpha', 'beta', 42],
        path: '/nodes/invokeai-node-pack',
      })
    ).toEqual({
      name: 'invokeai-node-pack',
      nodeCount: 3,
      nodeTypes: ['alpha', 'beta'],
      path: '/nodes/invokeai-node-pack',
    });
  });

  it('drops invalid packs and heals optional catalog fields', () => {
    expect(
      normalizeNodePackCatalog({
        custom_nodes_path: '/nodes',
        node_packs: [{ name: 'valid', path: '/nodes/valid' }, { name: 'missing-path' }, null],
      })
    ).toEqual({
      customNodesPath: '/nodes',
      nodePacks: [{ name: 'valid', nodeCount: 0, nodeTypes: [], path: '/nodes/valid' }],
    });
  });
});
