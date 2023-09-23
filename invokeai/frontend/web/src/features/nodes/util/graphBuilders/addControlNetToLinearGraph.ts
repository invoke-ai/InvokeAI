import { RootState } from 'app/store/store';
import { getValidControlNets } from 'features/controlNet/util/getValidControlNets';
import {
  CollectInvocation,
  ControlField,
  ControlNetInvocation,
} from 'services/api/types';
import { NonNullableGraph, zControlField } from '../../types/types';
import {
  CANVAS_COHERENCE_DENOISE_LATENTS,
  CONTROL_NET_COLLECT,
} from './constants';
import { addMainMetadata } from './metadata';

export const addControlNetToLinearGraph = (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): void => {
  const { isEnabled: isControlNetEnabled, controlNets } = state.controlNet;

  const validControlNets = getValidControlNets(controlNets);

  if (isControlNetEnabled && Boolean(validControlNets.length)) {
    if (validControlNets.length) {
      const controlnets: ControlField[] = [];
      // We have multiple controlnets, add ControlNet collector
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

      validControlNets.forEach((controlNet) => {
        const {
          controlNetId,
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
          id: `control_net_${controlNetId}`,
          type: 'controlnet',
          is_intermediate: true,
          begin_step_percent: beginStepPct,
          end_step_percent: endStepPct,
          control_mode: controlMode,
          resize_mode: resizeMode,
          control_model: model as ControlNetInvocation['control_model'],
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
          // Skip ControlNets without an unprocessed image - should never happen if everything is working correctly
          return;
        }

        graph.nodes[controlNetNode.id] = controlNetNode as ControlNetInvocation;

        controlnets.push(zControlField.parse(controlNetNode));

        graph.edges.push({
          source: { node_id: controlNetNode.id, field: 'control' },
          destination: {
            node_id: CONTROL_NET_COLLECT,
            field: 'item',
          },
        });

        if (CANVAS_COHERENCE_DENOISE_LATENTS in graph.nodes) {
          graph.edges.push({
            source: { node_id: controlNetNode.id, field: 'control' },
            destination: {
              node_id: CANVAS_COHERENCE_DENOISE_LATENTS,
              field: 'control',
            },
          });
        }
      });

      addMainMetadata(graph, { controlnets });
    }
  }
};
