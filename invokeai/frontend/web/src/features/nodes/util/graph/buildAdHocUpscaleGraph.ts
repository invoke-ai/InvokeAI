import type { RootState } from 'app/store/store';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import type { GraphType } from 'features/nodes/util/graph/generation/Graph';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getBoardField } from 'features/nodes/util/graph/graphBuilderUtils';
import type { ImageDTO } from 'services/api/types';
import { isSpandrelImageToImageModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { getModelMetadataField } from './canvas/metadata';
import { SPANDREL } from './constants';

type Arg = {
  image: ImageDTO;
  state: RootState;
};

export const buildAdHocUpscaleGraph = async ({ image, state }: Arg): Promise<GraphType> => {
  const { simpleUpscaleModel } = state.upscale;

  assert(simpleUpscaleModel, 'No upscale model found in state');

  const g = new Graph('adhoc-upscale-graph');
  g.addNode({
    id: SPANDREL,
    type: 'spandrel_image_to_image',
    image_to_image_model: simpleUpscaleModel,
    image,
    board: getBoardField(state),
    is_intermediate: false,
  });

  const modelConfig = await fetchModelConfigWithTypeGuard(simpleUpscaleModel.key, isSpandrelImageToImageModelConfig);

  g.upsertMetadata({
    upscale_model: getModelMetadataField(modelConfig),
  });

  return g.getGraph();
};
