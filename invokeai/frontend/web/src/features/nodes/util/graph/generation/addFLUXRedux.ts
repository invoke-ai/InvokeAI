import type { CanvasReferenceImageState, FLUXReduxConfig } from 'features/controlLayers/store/types';
import { isFLUXReduxConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';
import type { Invocation } from 'services/api/types';
import { assert } from 'tsafe';

type AddFLUXReduxResult = {
  addedFLUXReduxes: number;
};

type AddFLUXReduxArg = {
  entities: CanvasReferenceImageState[];
  g: Graph;
  collector: Invocation<'collect'>;
  model: ParameterModel;
};

export const addFLUXReduxes = ({ entities, g, collector, model }: AddFLUXReduxArg): AddFLUXReduxResult => {
  const validFLUXReduxes = entities
    .filter((entity) => entity.isEnabled)
    .filter((entity) => isFLUXReduxConfig(entity.ipAdapter))
    .filter((entity) => getGlobalReferenceImageWarnings(entity, model).length === 0);

  const result: AddFLUXReduxResult = {
    addedFLUXReduxes: 0,
  };

  for (const { id, ipAdapter } of validFLUXReduxes) {
    assert(isFLUXReduxConfig(ipAdapter), 'This should have been filtered out');
    result.addedFLUXReduxes++;

    addFLUXRedux(id, ipAdapter, g, collector);
  }

  return result;
};

const addFLUXRedux = (id: string, ipAdapter: FLUXReduxConfig, g: Graph, collector: Invocation<'collect'>) => {
  const { model: fluxReduxModel, image } = ipAdapter;
  assert(image, 'FLUX Redux image is required');
  assert(fluxReduxModel, 'FLUX Redux model is required');

  const node = g.addNode({
    id: `flux_redux_${id}`,
    type: 'flux_redux',
    redux_model: fluxReduxModel,
    image: {
      image_name: image.image_name,
    },
  });

  g.addEdge(node, 'redux_cond', collector, 'item');
};
