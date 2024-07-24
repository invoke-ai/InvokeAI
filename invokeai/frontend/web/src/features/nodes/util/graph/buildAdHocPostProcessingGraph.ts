import type { RootState } from 'app/store/store';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import type { GraphType } from 'features/nodes/util/graph/generation/Graph';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getBoardField } from 'features/nodes/util/graph/graphBuilderUtils';
import type { ImageDTO } from 'services/api/types';
import { isSpandrelImageToImageModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { SPANDREL } from './constants';

type Arg = {
  image: ImageDTO;
  state: RootState;
};

export const buildAdHocPostProcessingGraph = async ({ image, state }: Arg): Promise<GraphType> => {
  const { postProcessingModel } = state.upscale;

  assert(postProcessingModel, 'No post-processing model found in state');

  const g = new Graph('adhoc-post-processing-graph');
  g.addNode({
    id: SPANDREL,
    type: 'spandrel_image_to_image',
    image_to_image_model: postProcessingModel,
    image,
    board: getBoardField(state),
    is_intermediate: false,
  });

  const modelConfig = await fetchModelConfigWithTypeGuard(postProcessingModel.key, isSpandrelImageToImageModelConfig);

  g.upsertMetadata({
    upscale_model: Graph.getModelMetadataField(modelConfig),
  });

  return g.getGraph();
};
