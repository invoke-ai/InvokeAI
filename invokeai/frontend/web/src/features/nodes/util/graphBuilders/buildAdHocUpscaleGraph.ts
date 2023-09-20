import { NonNullableGraph } from 'features/nodes/types/types';
import { ESRGANModelName } from 'features/parameters/store/postprocessingSlice';
import {
  Graph,
  ESRGANInvocation,
  SaveImageInvocation,
} from 'services/api/types';
import { REALESRGAN as ESRGAN, SAVE_IMAGE } from './constants';

type Arg = {
  image_name: string;
  esrganModelName: ESRGANModelName;
};

export const buildAdHocUpscaleGraph = ({
  image_name,
  esrganModelName,
}: Arg): Graph => {
  const realesrganNode: ESRGANInvocation = {
    id: ESRGAN,
    type: 'esrgan',
    image: { image_name },
    model_name: esrganModelName,
    is_intermediate: true,
  };

  const saveImageNode: SaveImageInvocation = {
    id: SAVE_IMAGE,
    type: 'save_image',
    use_cache: false,
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
