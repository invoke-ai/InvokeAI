import { RootState } from 'app/store/store';
import { MetadataAccumulatorInvocation } from 'services/api/types';
import { NonNullableGraph } from '../../types/types';
import { METADATA_ACCUMULATOR, SDXL_TEXT_TO_LATENTS } from './constants';

export const addSDXLRefinerToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): void => {
  const { shouldUseSDXLRefiner, model } = state.generation;

  const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
    | MetadataAccumulatorInvocation
    | undefined;

  if (!shouldUseSDXLRefiner) return;

  // Unplug SDXL Text To Latents To Latents To Image
  graph.edges = graph.edges.filter(
    (e) =>
      !(
        e.source.node_id === SDXL_TEXT_TO_LATENTS &&
        ['latents'].includes(e.source.field)
      )
  );

  //   graph.nodes[SDXL_REFINER_MODEL_LOADER] = {
  //     id: SDXL_REFINER_MODEL_LOADER,
  //     type: 'sdxl_refiner_model_loader',
  //   };
};
