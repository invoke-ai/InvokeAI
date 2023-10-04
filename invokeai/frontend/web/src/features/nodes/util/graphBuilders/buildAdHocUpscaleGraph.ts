import { NonNullableGraph } from 'features/nodes/types/types';
import { ESRGANModelName } from 'features/parameters/store/postprocessingSlice';
import {
  Graph,
  ESRGANInvocation,
  SaveImageInvocation,
} from 'services/api/types';
import { REALESRGAN as ESRGAN, SAVE_IMAGE } from './constants';
import { BoardId } from 'features/gallery/store/types';

type Arg = {
  image_name: string;
  esrganModelName: ESRGANModelName;
  autoAddBoardId: BoardId;
};

export const buildAdHocUpscaleGraph = ({
  image_name,
  esrganModelName,
  autoAddBoardId,
}: Arg): Graph => {
  const realesrganNode: ESRGANInvocation = {
    id: ESRGAN,
    type: 'esrgan',
    image: { image_name },
    model_name: esrganModelName,
    tile_size: 512,
    is_intermediate: true,
  };

  const saveImageNode: SaveImageInvocation = {
    id: SAVE_IMAGE,
    type: 'save_image',
    use_cache: false,
    is_intermediate: false,
    board: autoAddBoardId === 'none' ? undefined : { board_id: autoAddBoardId },
  };

  const graph: NonNullableGraph = {
    id: `adhoc-esrgan-graph`,
    nodes: {
      [ESRGAN]: realesrganNode,
      [SAVE_IMAGE]: saveImageNode,
    },
    edges: [
      {
        source: {
          node_id: ESRGAN,
          field: 'image',
        },
        destination: {
          node_id: SAVE_IMAGE,
          field: 'image',
        },
      },
    ],
  };

  return graph;
};
