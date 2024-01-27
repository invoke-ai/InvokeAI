import type { BoardId } from 'features/gallery/store/types';
import type { ParamESRGANModelName } from 'features/parameters/store/postprocessingSlice';
import type { ESRGANInvocation, Graph, LinearUIOutputInvocation, NonNullableGraph } from 'services/api/types';

import { ESRGAN, LINEAR_UI_OUTPUT } from './constants';
import { addCoreMetadataNode, upsertMetadata } from './metadata';

type Arg = {
  image_name: string;
  esrganModelName: ParamESRGANModelName;
  autoAddBoardId: BoardId;
};

export const buildAdHocUpscaleGraph = ({ image_name, esrganModelName, autoAddBoardId }: Arg): Graph => {
  const realesrganNode: ESRGANInvocation = {
    id: ESRGAN,
    type: 'esrgan',
    image: { image_name },
    model_name: esrganModelName,
    is_intermediate: true,
  };

  const linearUIOutputNode: LinearUIOutputInvocation = {
    id: LINEAR_UI_OUTPUT,
    type: 'linear_ui_output',
    use_cache: false,
    is_intermediate: false,
    board: autoAddBoardId === 'none' ? undefined : { board_id: autoAddBoardId },
  };

  const graph: NonNullableGraph = {
    id: `adhoc-esrgan-graph`,
    nodes: {
      [ESRGAN]: realesrganNode,
      [LINEAR_UI_OUTPUT]: linearUIOutputNode,
    },
    edges: [
      {
        source: {
          node_id: ESRGAN,
          field: 'image',
        },
        destination: {
          node_id: LINEAR_UI_OUTPUT,
          field: 'image',
        },
      },
    ],
  };

  addCoreMetadataNode(graph, {}, ESRGAN);
  upsertMetadata(graph, {
    esrgan_model: esrganModelName,
  });

  return graph;
};
