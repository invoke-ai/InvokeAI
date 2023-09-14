import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { isInvocationNode } from 'features/nodes/types/types';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import i18n from 'i18next';
import { forEach, map } from 'lodash-es';
import { getConnectedEdges } from 'reactflow';

const selector = createSelector(
  [stateSelector, activeTabNameSelector],
  (state, activeTabName) => {
    const { generation, system, nodes } = state;
    const { initialImage, model } = generation;

    const { isProcessing, isConnected } = system;

    const reasons: string[] = [];

    // Cannot generate if not connected
    if (!isConnected) {
      reasons.push(i18n.t('parameters.invoke.systemDisconnected'));
    }

    if (activeTabName === 'img2img' && !initialImage) {
      reasons.push(i18n.t('parameters.invoke.noInitialImageSelected'));
    }

    if (activeTabName === 'nodes') {
      if (nodes.shouldValidateGraph) {
        if (!nodes.nodes.length) {
          reasons.push(i18n.t('parameters.invoke.noNodesInGraph'));
        }

        nodes.nodes.forEach((node) => {
          if (!isInvocationNode(node)) {
            return;
          }

          const nodeTemplate = nodes.nodeTemplates[node.data.type];

          if (!nodeTemplate) {
            // Node type not found
            reasons.push(i18n.t('parameters.invoke.missingNodeTemplate'));
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
              reasons.push(i18n.t('parameters.invoke.missingFieldTemplate'));
              return;
            }

            if (
              fieldTemplate.required &&
              field.value === undefined &&
              !hasConnection
            ) {
              reasons.push(
                i18n.t('parameters.invoke.missingInputForField', {
                  nodeLabel: node.data.label || nodeTemplate.title,
                  fieldLabel: field.label || fieldTemplate.title,
                })
              );
              return;
            }
          });
        });
      }
    } else {
      if (!model) {
        reasons.push(i18n.t('parameters.invoke.noModelSelected'));
      }

      if (state.controlNet.isEnabled) {
        map(state.controlNet.controlNets).forEach((controlNet, i) => {
          if (!controlNet.isEnabled) {
            return;
          }
          if (!controlNet.model) {
            reasons.push(
              i18n.t('parameters.invoke.noModelForControlNet', { index: i + 1 })
            );
          }

          if (
            !controlNet.controlImage ||
            (!controlNet.processedControlImage &&
              controlNet.processorType !== 'none')
          ) {
            reasons.push(
              i18n.t('parameters.invoke.noControlImageForControlNet', {
                index: i + 1,
              })
            );
          }
        });
      }
    }

    return { isReady: !reasons.length, isProcessing, reasons };
  },
  defaultSelectorOptions
);

export const useIsReadyToEnqueue = () => {
  const { isReady, isProcessing, reasons } = useAppSelector(selector);
  return { isReady, isProcessing, reasons };
};
