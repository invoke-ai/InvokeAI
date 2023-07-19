import { NonNullableGraph } from 'features/nodes/types/types';
import { ESRGANModelName } from 'features/parameters/store/postprocessingSlice';
import { Graph, ESRGANInvocation } from 'services/api/types';
import { REALESRGAN as ESRGAN } from './constants';

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
    is_intermediate: false,
  };

  const graph: NonNullableGraph = {
    id: `adhoc-esrgan-graph`,
    nodes: {
      [ESRGAN]: realesrganNode,
    },
    edges: [],
  };

  return graph;
};
