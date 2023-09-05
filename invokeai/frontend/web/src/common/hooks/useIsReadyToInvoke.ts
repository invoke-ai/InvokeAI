import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { isInvocationNode } from 'features/nodes/types/types';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { forEach, map } from 'lodash-es';
import { getConnectedEdges } from 'reactflow';

const selector = createSelector(
  [stateSelector, activeTabNameSelector],
  (state, activeTabName) => {
    const { generation, system, nodes } = state;
    const { initialImage, model } = generation;

    const { isProcessing, isConnected } = system;

    const reasons: string[] = [];

    // Cannot generate if already processing an image
    if (isProcessing) {
      reasons.push('System busy');
    }

    // Cannot generate if not connected
    if (!isConnected) {
      reasons.push('System disconnected');
    }

    if (activeTabName === 'img2img' && !initialImage) {
      reasons.push('No initial image selected');
    }

    if (activeTabName === 'nodes') {
      if (nodes.shouldValidateGraph) {
        if (!nodes.nodes.length) {
          reasons.push('No nodes in graph');
        }

        nodes.nodes.forEach((node) => {
          if (!isInvocationNode(node)) {
            return;
          }

          const nodeTemplate = nodes.nodeTemplates[node.data.type];

          if (!nodeTemplate) {
            // Node type not found
            reasons.push('Missing node template');
            return;
          }

          const connectedEdges = getConnectedEdges([node], nodes.edges);

          forEach(node.data.inputs, (field) => {
            const fieldTemplate = nodeTemplate.inputs[field.name];
            const hasConnection = connectedEdges.some(
              (edge) =>
                edge.target === node.id && edge.targetHandle === field.name
            );

            if (!fieldTemplate) {
              reasons.push('Missing field template');
              return;
            }

            if (
              fieldTemplate.required &&
              field.value === undefined &&
              !hasConnection
            ) {
              reasons.push(
                `${node.data.label || nodeTemplate.title} -> ${
                  field.label || fieldTemplate.title
                } missing input`
              );
              return;
            }
          });
        });
      }
    } else {
      if (!model) {
        reasons.push('No model selected');
      }

      if (state.controlNet.isEnabled) {
        map(state.controlNet.controlNets).forEach((controlNet, i) => {
          if (!controlNet.isEnabled) {
            return;
          }
          if (!controlNet.model) {
            reasons.push(`ControlNet ${i + 1} has no model selected.`);
          }

          if (
            !controlNet.controlImage ||
            (!controlNet.processedControlImage &&
              controlNet.processorType !== 'none')
          ) {
            reasons.push(`ControlNet ${i + 1} has no control image`);
          }
        });
      }
    }

    return { isReady: !reasons.length, isProcessing, reasons };
  },
  defaultSelectorOptions
);

export const useIsReadyToInvoke = () => {
  const { isReady, isProcessing, reasons } = useAppSelector(selector);
  return { isReady, isProcessing, reasons };
};
