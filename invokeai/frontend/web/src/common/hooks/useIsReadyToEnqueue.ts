import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterAll,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { isControlNetOrT2IAdapter } from 'features/controlAdapters/store/types';
import { selectControlLayersSlice } from 'features/controlLayers/store/controlLayersSlice';
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
import { assert } from 'tsafe';

const selector = createMemoizedSelector(
  [
    selectControlAdaptersSlice,
    selectGenerationSlice,
    selectSystemSlice,
    selectNodesSlice,
    selectDynamicPromptsSlice,
    selectControlLayersSlice,
    activeTabNameSelector,
  ],
  (controlAdapters, generation, system, nodes, dynamicPrompts, controlLayers, activeTabName) => {
    const { initialImage, model } = generation;
    const { positivePrompt } = controlLayers.present;

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

      if (activeTabName === 'txt2img') {
        // Handling for Control Layers - only exists on txt2img tab now
        controlLayers.present.layers
          .filter((l) => l.isEnabled)
          .flatMap((l) => {
            if (l.type === 'control_adapter_layer') {
              return l.controlAdapter;
            } else if (l.type === 'ip_adapter_layer') {
              return l.ipAdapter;
            } else if (l.type === 'regional_guidance_layer') {
              return l.ipAdapters;
            }
            assert(false);
          })
          .forEach((ca, i) => {
            const hasNoModel = !ca.model;
            const mismatchedModelBase = ca.model?.base !== model?.base;
            const hasNoImage = !ca.image;
            const imageNotProcessed =
              (ca.type === 'controlnet' || ca.type === 't2i_adapter') && !ca.processedImage && ca.processorConfig;

            if (hasNoModel) {
              reasons.push(
                i18n.t('parameters.invoke.noModelForControlAdapter', {
                  number: i + 1,
                })
              );
            }
            if (mismatchedModelBase) {
              // This should never happen, just a sanity check
              reasons.push(
                i18n.t('parameters.invoke.incompatibleBaseModelForControlAdapter', {
                  number: i + 1,
                })
              );
            }
            if (hasNoImage) {
              reasons.push(
                i18n.t('parameters.invoke.noControlImageForControlAdapter', {
                  number: i + 1,
                })
              );
            }
            if (imageNotProcessed) {
              reasons.push(
                i18n.t('parameters.invoke.imageNotProcessedForControlAdapter', {
                  number: i + 1,
                })
              );
            }
          });
      } else {
        // Handling for all other tabs
        selectControlAdapterAll(controlAdapters)
          .filter((ca) => ca.isEnabled)
          .forEach((ca, i) => {
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
    }

    return { isReady: !reasons.length, reasons };
  }
);

export const useIsReadyToEnqueue = () => {
  const { isReady, reasons } = useAppSelector(selector);
  return { isReady, reasons };
};
