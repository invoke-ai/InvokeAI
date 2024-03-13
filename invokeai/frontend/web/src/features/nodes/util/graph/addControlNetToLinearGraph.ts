import type { RootState } from 'app/store/store';
import { selectValidControlNets } from 'features/controlAdapters/store/controlAdaptersSlice';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import type {
  CollectInvocation,
  ControlNetInvocation,
  CoreMetadataInvocation,
  NonNullableGraph,
} from 'services/api/types';
import { isControlNetModelConfig } from 'services/api/types';

import { CONTROL_NET_COLLECT } from './constants';
import { getModelMetadataField, upsertMetadata } from './metadata';

export const addControlNetToLinearGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): Promise<void> => {
  const validControlNets = selectValidControlNets(state.controlAdapters).filter(
    ({ model, processedControlImage, processorType, controlImage, isEnabled }) => {
      const hasModel = Boolean(model);
      const doesBaseMatch = model?.base === state.generation.model?.base;
      const hasControlImage = (processedControlImage && processorType !== 'none') || controlImage;

      return isEnabled && hasModel && doesBaseMatch && hasControlImage;
    }
  );

  // const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
  //   | MetadataAccumulatorInvocation
  //   | undefined;

  const controlNetMetadata: CoreMetadataInvocation['controlnets'] = [];

  if (validControlNets.length) {
    // Even though denoise_latents' control input is collection or scalar, keep it simple and always use a collect
    const controlNetIterateNode: CollectInvocation = {
      id: CONTROL_NET_COLLECT,
      type: 'collect',
      is_intermediate: true,
    };
    graph.nodes[CONTROL_NET_COLLECT] = controlNetIterateNode;
    graph.edges.push({
      source: { node_id: CONTROL_NET_COLLECT, field: 'collection' },
      destination: {
        node_id: baseNodeId,
        field: 'control',
      },
    });

    for (const controlNet of validControlNets) {
      if (!controlNet.model) {
        return;
      }
      const {
        id,
        controlImage,
        processedControlImage,
        beginStepPct,
        endStepPct,
        controlMode,
        resizeMode,
        model,
        processorType,
        weight,
      } = controlNet;

      const controlNetNode: ControlNetInvocation = {
        id: `control_net_${id}`,
        type: 'controlnet',
        is_intermediate: true,
        begin_step_percent: beginStepPct,
        end_step_percent: endStepPct,
        control_mode: controlMode,
        resize_mode: resizeMode,
        control_model: model,
        control_weight: weight,
      };

      if (processedControlImage && processorType !== 'none') {
        // We've already processed the image in the app, so we can just use the processed image
        controlNetNode.image = {
          image_name: processedControlImage,
        };
      } else if (controlImage) {
        // The control image is preprocessed
        controlNetNode.image = {
          image_name: controlImage,
        };
      } else {
        // Skip CAs without an unprocessed image - should never happen, we already filtered the list of valid CAs
        return;
      }

      graph.nodes[controlNetNode.id] = controlNetNode as ControlNetInvocation;

      const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isControlNetModelConfig);

      controlNetMetadata.push({
        control_model: getModelMetadataField(modelConfig),
        control_weight: weight,
        control_mode: controlMode,
        begin_step_percent: beginStepPct,
        end_step_percent: endStepPct,
        resize_mode: resizeMode,
        image: controlNetNode.image,
      });

      graph.edges.push({
        source: { node_id: controlNetNode.id, field: 'control' },
        destination: {
          node_id: CONTROL_NET_COLLECT,
          field: 'item',
        },
      });
    }
    upsertMetadata(graph, { controlnets: controlNetMetadata });
  }
};
