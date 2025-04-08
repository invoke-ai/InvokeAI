import type {
  CanvasReferenceImageState,
  FLUXReduxConfig,
  FLUXReduxImageInfluence,
} from 'features/controlLayers/store/types';
import { isFLUXReduxConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, MainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

type AddFLUXReduxResult = {
  addedFLUXReduxes: number;
};

type AddFLUXReduxArg = {
  entities: CanvasReferenceImageState[];
  g: Graph;
  collector: Invocation<'collect'>;
  model: MainModelConfig;
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

/**
 * To fine-tune the image influence, edit this object.
 * - downsampling_factor: 1 to 9, where 1 is the most image influence and 9 is the least. 1 is FLUX redux in its original form.
 * - downsampling_function: the function used to downsample the image. Defaults to 'area'. Dunno about how it affects the image.
 * - weight: 0 to 1. the conditioning is multiplied by the square of this value. 1 means no change.
 *
 * See invokeai/app/invocations/flux_redux.py for more details.
 */
export const IMAGE_INFLUENCE_TO_SETTINGS: Record<
  FLUXReduxImageInfluence,
  Pick<Invocation<'flux_redux'>, 'downsampling_factor' | 'downsampling_function' | 'weight'>
> = {
  lowest: {
    downsampling_factor: 5,
    // downsampling_function: 'area',
    weight: 1,
  },
  low: {
    downsampling_factor: 4,
    // downsampling_function: 'area',
    weight: 1,
  },
  medium: {
    downsampling_factor: 3,
    // downsampling_function: 'area',
    weight: 1,
  },
  high: {
    downsampling_factor: 2,
    // downsampling_function: 'area',
    weight: 1,
  },
  highest: {
    downsampling_factor: 1,
    // downsampling_function: 'area',
    weight: 1,
  },
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
    ...IMAGE_INFLUENCE_TO_SETTINGS[ipAdapter.imageInfluence ?? 'highest'],
  });

  g.addEdge(node, 'redux_cond', collector, 'item');
};
