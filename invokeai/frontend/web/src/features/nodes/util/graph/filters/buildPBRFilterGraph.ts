import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { GraphType } from 'features/nodes/util/graph/generation/Graph';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getBoardField } from 'features/nodes/util/graph/graphBuilderUtils';
import type { ImageDTO } from 'services/api/types';

type Arg = {
  image: ImageDTO;
  state: RootState;
};

export const buildPBRFilterGraph = async ({ image, state }: Arg): Promise<GraphType> => {
  const g = new Graph('pbr-maps-graph');
  g.addNode({
    type: 'pbr_maps',
    id: getPrefixedId('pbr_maps'),
    tile_size: 512,
    border_mode: 'none',
    image,
    board: getBoardField(state),
    is_intermediate: false,
  });

  return await g.getGraph();
};
