import type { RootState } from 'app/store/store';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import {
  type ImageDTO,
  type Invocation,
  isSpandrelImageToImageModelConfig,
  type NonNullableGraph,
} from 'services/api/types';
import { assert } from 'tsafe';

import { addCoreMetadataNode, getModelMetadataField, upsertMetadata } from './canvas/metadata';
import { SPANDREL } from './constants';

type Arg = {
  image: ImageDTO;
  state: RootState;
};

export const buildAdHocUpscaleGraph = async ({ image, state }: Arg): Promise<NonNullableGraph> => {
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
  const modelConfig = await fetchModelConfigWithTypeGuard(simpleUpscaleModel.key, isSpandrelImageToImageModelConfig);

  addCoreMetadataNode(graph, {}, SPANDREL);
  upsertMetadata(graph, {
    upscale_model: getModelMetadataField(modelConfig),
  });

  return graph;
};
