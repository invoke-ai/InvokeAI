import type { RootState } from 'app/store/store';
import type { Graph, ImageDTO, Invocation, NonNullableGraph } from 'services/api/types';
import { assert } from 'tsafe';

import { addCoreMetadataNode, upsertMetadata } from './canvas/metadata';
import { SPANDREL } from './constants';

type Arg = {
  image: ImageDTO;
  state: RootState;
};

export const buildAdHocUpscaleGraph = ({ image, state }: Arg): Graph => {
  const { simpleUpscaleModel } = state.upscale;

  assert(simpleUpscaleModel, 'No upscale model found in state');

  const upscaleNode: Invocation<'spandrel_image_to_image'> = {
    id: SPANDREL,
    type: 'spandrel_image_to_image',
    image_to_image_model: simpleUpscaleModel,
    tile_size: 500,
    image,
  };

  const graph: NonNullableGraph = {
    id: `adhoc-upscale-graph`,
    nodes: {
      [SPANDREL]: upscaleNode,
    },
    edges: [],
  };

  addCoreMetadataNode(graph, {}, SPANDREL);
  upsertMetadata(graph, {
    spandrel_model: simpleUpscaleModel,
  });

  return graph;
};
