import { describe, expect, it } from 'vitest';

import { buildPbrMapsGraph, PBR_MAPS_NODE_ID } from './pbrMapsGraph';

describe('buildPbrMapsGraph', () => {
  it('leaves the maps Uncategorized when no board is auto-adding', () => {
    for (const boardId of [null, 'none']) {
      expect(buildPbrMapsGraph({ boardId, imageName: 'texture.png' }).nodes[PBR_MAPS_NODE_ID]).not.toHaveProperty(
        'board'
      );
    }
  });
});
