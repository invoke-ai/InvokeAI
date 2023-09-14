import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { unset } from 'lodash-es';
import { RandomIntInvocation } from 'services/api/types';
import {
  CANVAS_COHERENCE_NOISE,
  METADATA_ACCUMULATOR,
  NOISE,
  RANDOM_INT,
} from './constants';

export const addRandomizeSeedToLinearGraph = (
  state: RootState,
  graph: NonNullableGraph
) => {
  const { shouldRandomizeSeed } = state.generation;
  const hasMetadataAccumulator = METADATA_ACCUMULATOR in graph.nodes;

  if (!shouldRandomizeSeed) {
    return;
  }

  const randomIntNode: RandomIntInvocation = {
    id: RANDOM_INT,
    type: 'rand_int',
    is_intermediate: true,
  };
  graph.nodes[RANDOM_INT] = randomIntNode;
  graph.edges.push({
    source: { node_id: RANDOM_INT, field: 'value' },
    destination: { node_id: NOISE, field: 'seed' },
  });
  hasMetadataAccumulator &&
    graph.edges.push({
      source: { node_id: RANDOM_INT, field: 'value' },
      destination: { node_id: METADATA_ACCUMULATOR, field: 'seed' },
    });
  if (CANVAS_COHERENCE_NOISE in graph.nodes) {
    graph.edges.push({
      source: { node_id: RANDOM_INT, field: 'value' },
      destination: { node_id: CANVAS_COHERENCE_NOISE, field: 'seed' },
    });
  }

  unset(graph.nodes[NOISE], 'seed');
};
