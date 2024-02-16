import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterAll,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { isControlNetOrT2IAdapter } from 'features/controlAdapters/store/types';
import { selectDynamicPromptsSlice } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { selectSystemSlice } from 'features/system/store/systemSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import i18n from 'i18next';
import { forEach } from 'lodash-es';
import { getConnectedEdges } from 'reactflow';

const selector = createMemoizedSelector(
  [
    selectControlAdaptersSlice,
    selectGenerationSlice,
    selectSystemSlice,
    selectNodesSlice,
    selectDynamicPromptsSlice,
    activeTabNameSelector,
  ],
  (controlAdapters, generation, system, nodes, dynamicPrompts, activeTabName) => {
    const { initialImage, model, positivePrompt } = generation;

    const { isConnected } = system;

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

          const nodeTemplate = nodes.templates[node.data.type];

          if (!nodeTemplate) {
            // Node type not found
            reasons.push(i18n.t('parameters.invoke.missingNodeTemplate'));
            return;
          }

          const connectedEdges = getConnectedEdges([node], nodes.edges);

          forEach(node.data.inputs, (field) => {
            const fieldTemplate = nodeTemplate.inputs[field.name];
            const hasConnection = connectedEdges.some(
              (edge) => edge.target === node.id && edge.targetHandle === field.name
            );

            if (!fieldTemplate) {
              reasons.push(i18n.t('parameters.invoke.missingFieldTemplate'));
              return;
            }

            if (fieldTemplate.required && field.value === undefined && !hasConnection) {
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
      if (dynamicPrompts.prompts.length === 0 && getShouldProcessPrompt(positivePrompt)) {
        reasons.push(i18n.t('parameters.invoke.noPrompts'));
      }

      if (!model) {
        reasons.push(i18n.t('parameters.invoke.noModelSelected'));
      }

      selectControlAdapterAll(controlAdapters).forEach((ca, i) => {
        if (!ca.isEnabled) {
          return;
        }

        if (!ca.model) {
          reasons.push(
            i18n.t('parameters.invoke.noModelForControlAdapter', {
              number: i + 1,
            })
          );
        } else if (ca.model.base !== model?.base) {
          // This should never happen, just a sanity check
          reasons.push(
            i18n.t('parameters.invoke.incompatibleBaseModelForControlAdapter', {
              number: i + 1,
            })
          );
        }

        if (
          !ca.controlImage ||
          (isControlNetOrT2IAdapter(ca) && !ca.processedControlImage && ca.processorType !== 'none')
        ) {
          reasons.push(
            i18n.t('parameters.invoke.noControlImageForControlAdapter', {
              number: i + 1,
            })
          );
        }
      });
    }

    return { isReady: !reasons.length, reasons };
  }
);

export const useIsReadyToEnqueue = () => {
  const { isReady, reasons } = useAppSelector(selector);
  return { isReady, reasons };
};
