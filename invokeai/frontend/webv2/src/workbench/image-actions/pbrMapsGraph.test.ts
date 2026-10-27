import { describe, expect, it } from 'vitest';

import { buildPbrMapsGraph, PBR_MAPS_NODE_ID } from './pbrMapsGraph';

describe('buildPbrMapsGraph', () => {
  it('emits one durable pbr_maps node over the image, destined for the given board', () => {
    const graph = buildPbrMapsGraph({ boardId: 'board-7', imageName: 'texture.png' });

    expect(graph.edges).toEqual([]);
    expect(Object.keys(graph.nodes)).toEqual([PBR_MAPS_NODE_ID]);
    expect(graph.nodes[PBR_MAPS_NODE_ID]).toEqual({
      board: { board_id: 'board-7' },
      border_mode: 'none',
      id: PBR_MAPS_NODE_ID,
      image: { image_name: 'texture.png' },
      is_intermediate: false,
      tile_size: 512,
      type: 'pbr_maps',
      use_cache: false,
    });
  });

  it('leaves the maps Uncategorized when no board is auto-adding', () => {
    for (const boardId of [null, 'none']) {
      expect(buildPbrMapsGraph({ boardId, imageName: 'texture.png' }).nodes[PBR_MAPS_NODE_ID]).not.toHaveProperty(
        'board'
      );
    }
  });
});
