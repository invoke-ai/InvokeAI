import { RootState } from 'app/store/store';
import { forEach, size } from 'lodash-es';
import { CollectInvocation, ControlNetInvocation } from 'services/api';
import { NonNullableGraph } from '../types/types';

const CONTROL_NET_COLLECT = 'control_net_collect';

export const addControlNetToLinearGraph = (
  graph: NonNullableGraph,
  baseNodeId: string,
  state: RootState
): void => {
  const { isEnabled: isControlNetEnabled, controlNets } = state.controlNet;

  // Add ControlNet
  if (isControlNetEnabled) {
    if (size(controlNets) > 1) {
      const controlNetIterateNode: CollectInvocation = {
        id: CONTROL_NET_COLLECT,
        type: 'collect',
      };
      graph.nodes[controlNetIterateNode.id] = controlNetIterateNode;
      graph.edges.push({
        source: { node_id: controlNetIterateNode.id, field: 'collection' },
        destination: {
          node_id: baseNodeId,
          field: 'control',
        },
      });
    }

    forEach(controlNets, (controlNet, index) => {
      const {
        controlNetId,
        isEnabled,
        controlImage,
        processedControlImage,
        beginStepPct,
        endStepPct,
        model,
        processorType,
        weight,
      } = controlNet;

      if (!isEnabled) {
        // Skip disabled ControlNets
        return;
      }

      const controlNetNode: ControlNetInvocation = {
        id: `control_net_${controlNetId}`,
        type: 'controlnet',
        begin_step_percent: beginStepPct,
        end_step_percent: endStepPct,
        control_model: model as ControlNetInvocation['control_model'],
        control_weight: weight,
      };

      if (processedControlImage && processorType !== 'none') {
        // We've already processed the image in the app, so we can just use the processed image
        const { image_name, image_origin } = processedControlImage;
        controlNetNode.image = {
          image_name,
          image_origin,
        };
      } else if (controlImage) {
        // The control image is preprocessed
        const { image_name, image_origin } = controlImage;
        controlNetNode.image = {
          image_name,
          image_origin,
        };
      } else {
        // Skip ControlNets without an unprocessed image - should never happen if everything is working correctly
        return;
      }

      graph.nodes[controlNetNode.id] = controlNetNode;

      if (size(controlNets) > 1) {
        graph.edges.push({
          source: { node_id: controlNetNode.id, field: 'control' },
          destination: {
            node_id: CONTROL_NET_COLLECT,
            field: 'item',
          },
        });
      } else {
        graph.edges.push({
          source: { node_id: controlNetNode.id, field: 'control' },
          destination: {
            node_id: baseNodeId,
            field: 'control',
          },
        });
      }
    });
  }
};
