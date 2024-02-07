import type { RootState } from 'app/store/store';
import { getBoardField, getIsIntermediate } from 'features/nodes/util/graph/graphBuilderUtils';
import type { ESRGANInvocation, Graph, NonNullableGraph } from 'services/api/types';

import { ESRGAN } from './constants';
import { addCoreMetadataNode, upsertMetadata } from './metadata';

type Arg = {
  image_name: string;
  state: RootState;
};

export const buildAdHocUpscaleGraph = ({ image_name, state }: Arg): Graph => {
  const { esrganModelName } = state.postprocessing;

  const realesrganNode: ESRGANInvocation = {
    id: ESRGAN,
    type: 'esrgan',
    image: { image_name },
    model_name: esrganModelName,
    is_intermediate: getIsIntermediate(state),
    board: getBoardField(state),
  };

  const graph: NonNullableGraph = {
    id: `adhoc-esrgan-graph`,
    nodes: {
      [ESRGAN]: realesrganNode,
    },
    edges: [],
  };

  addCoreMetadataNode(graph, {}, ESRGAN);
  upsertMetadata(graph, {
    esrgan_model: esrganModelName,
  });

  return graph;
};
