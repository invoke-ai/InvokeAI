import { RootState } from 'app/store/store';
import { filter, forEach, size } from 'lodash-es';
import { CollectInvocation, ControlNetInvocation } from 'services/api/types';
import { NonNullableGraph } from '../types/types';
import { CONTROL_NET_COLLECT } from './graphBuilders/constants';

export const addControlNetToLinearGraph = (
  graph: NonNullableGraph,
  baseNodeId: string,
  state: RootState
): void => {
  const { isEnabled: isControlNetEnabled, controlNets } = state.controlNet;

  const validControlNets = filter(
    controlNets,
    (c) =>
      c.isEnabled &&
      (Boolean(c.processedControlImage) ||
        (c.processorType === 'none' && Boolean(c.controlImage)))
  );

  // Add ControlNet
  if (isControlNetEnabled && validControlNets.length > 0) {
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

    forEach(controlNets, (controlNet) => {
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
