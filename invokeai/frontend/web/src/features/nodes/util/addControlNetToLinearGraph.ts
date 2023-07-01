import { RootState } from 'app/store/store';
import { filter } from 'lodash-es';
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

  if (isControlNetEnabled && Boolean(validControlNets.length)) {
    if (validControlNets.length > 1) {
      // We have multiple controlnets, add ControlNet collector
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

    validControlNets.forEach((controlNet) => {
      const {
        controlNetId,
        controlImage,
        processedControlImage,
        beginStepPct,
        endStepPct,
        controlMode,
        model,
        processorType,
        weight,
      } = controlNet;

      const controlNetNode: ControlNetInvocation = {
        id: `control_net_${controlNetId}`,
        type: 'controlnet',
        begin_step_percent: beginStepPct,
        end_step_percent: endStepPct,
        control_mode: controlMode,
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

      if (validControlNets.length > 1) {
        // if we have multiple controlnets, link to the collector
        graph.edges.push({
          source: { node_id: controlNetNode.id, field: 'control' },
          destination: {
            node_id: CONTROL_NET_COLLECT,
            field: 'item',
          },
        });
      } else {
        // otherwise, link directly to the base node
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
