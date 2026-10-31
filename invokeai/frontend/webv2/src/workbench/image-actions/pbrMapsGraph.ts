import type { QueueBackendGraph } from '@features/queue/contracts';

export const PBR_MAPS_NODE_ID = 'pbr_maps';

/**
 * One `pbr_maps` node over a gallery image: whole-image tiles with no seam handling, saved to the board as the
 * normal, roughness and displacement maps. `boardId` null or `'none'` leaves them Uncategorized.
 */
export const buildPbrMapsGraph = ({
  boardId,
  imageName,
}: {
  boardId: string | null;
  imageName: string;
}): QueueBackendGraph => ({
  edges: [],
  id: 'pbr-maps-graph',
  nodes: {
    [PBR_MAPS_NODE_ID]: {
      id: PBR_MAPS_NODE_ID,
      type: 'pbr_maps',
      ...(boardId !== null && boardId !== 'none' ? { board: { board_id: boardId } } : {}),
      border_mode: 'none',
      image: { image_name: imageName },
      is_intermediate: false,
      tile_size: 512,
      use_cache: false,
    },
  },
});
