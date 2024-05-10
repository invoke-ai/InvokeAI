import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectControlAdapterAll,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { isControlNetOrT2IAdapter } from 'features/controlAdapters/store/types';
import { selectControlLayersSlice } from 'features/controlLayers/store/controlLayersSlice';
import type { Layer } from 'features/controlLayers/store/types';
import { selectDynamicPromptsSlice } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { selectSystemSlice } from 'features/system/store/systemSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import i18n from 'i18next';
import { forEach, upperFirst } from 'lodash-es';
import { getConnectedEdges } from 'reactflow';

const LAYER_TYPE_TO_TKEY: Record<Layer['type'], string> = {
  initial_image_layer: 'controlLayers.globalInitialImage',
  control_adapter_layer: 'controlLayers.globalControlAdapter',
  ip_adapter_layer: 'controlLayers.globalIPAdapter',
  regional_guidance_layer: 'controlLayers.regionalGuidance',
};

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
    const { model } = generation;
    const { size } = controlLayers.present;
    const { positivePrompt } = controlLayers.present;

    const { isConnected } = system;

    const reasons: { prefix?: string; content: string }[] = [];

    // Cannot generate if not connected
    if (!isConnected) {
      reasons.push({ content: i18n.t('parameters.invoke.systemDisconnected') });
    }

    if (activeTabName === 'workflows') {
      if (nodes.shouldValidateGraph) {
        if (!nodes.nodes.length) {
          reasons.push({ content: i18n.t('parameters.invoke.noNodesInGraph') });
        }

        nodes.nodes.forEach((node) => {
          if (!isInvocationNode(node)) {
            return;
          }

          const nodeTemplate = nodes.templates[node.data.type];

          if (!nodeTemplate) {
            // Node type not found
            reasons.push({ content: i18n.t('parameters.invoke.missingNodeTemplate') });
            return;
          }

          const connectedEdges = getConnectedEdges([node], nodes.edges);

          forEach(node.data.inputs, (field) => {
            const fieldTemplate = nodeTemplate.inputs[field.name];
            const hasConnection = connectedEdges.some(
              (edge) => edge.target === node.id && edge.targetHandle === field.name
            );

            if (!fieldTemplate) {
              reasons.push({ content: i18n.t('parameters.invoke.missingFieldTemplate') });
              return;
            }

            if (fieldTemplate.required && field.value === undefined && !hasConnection) {
              reasons.push({
                content: i18n.t('parameters.invoke.missingInputForField', {
                  nodeLabel: node.data.label || nodeTemplate.title,
                  fieldLabel: field.label || fieldTemplate.title,
                }),
              });
              return;
            }
          });
        });
      }
    } else {
      if (dynamicPrompts.prompts.length === 0 && getShouldProcessPrompt(positivePrompt)) {
        reasons.push({ content: i18n.t('parameters.invoke.noPrompts') });
      }

      if (!model) {
        reasons.push({ content: i18n.t('parameters.invoke.noModelSelected') });
      }

      if (activeTabName === 'generation') {
        // Handling for generation tab
        controlLayers.present.layers
          .filter((l) => l.isEnabled)
          .forEach((l, i) => {
            const layerLiteral = i18n.t('controlLayers.layers_one');
            const layerNumber = i + 1;
            const layerType = i18n.t(LAYER_TYPE_TO_TKEY[l.type]);
            const prefix = `${layerLiteral} #${layerNumber} (${layerType})`;
            const problems: string[] = [];
            if (l.type === 'control_adapter_layer') {
              // Must have model
              if (!l.controlAdapter.model) {
                problems.push(i18n.t('parameters.invoke.layer.controlAdapterNoModelSelected'));
              }
              // Model base must match
              if (l.controlAdapter.model?.base !== model?.base) {
                problems.push(i18n.t('parameters.invoke.layer.controlAdapterIncompatibleBaseModel'));
              }
              // Must have a control image OR, if it has a processor, it must have a processed image
              if (!l.controlAdapter.image) {
                problems.push(i18n.t('parameters.invoke.layer.controlAdapterNoImageSelected'));
              } else if (l.controlAdapter.processorConfig && !l.controlAdapter.processedImage) {
                problems.push(i18n.t('parameters.invoke.layer.controlAdapterImageNotProcessed'));
              }
              // T2I Adapters require images have dimensions that are multiples of 64
              if (l.controlAdapter.type === 't2i_adapter' && (size.width % 64 !== 0 || size.height % 64 !== 0)) {
                problems.push(i18n.t('parameters.invoke.layer.t2iAdapterIncompatibleDimensions'));
              }
            }

            if (l.type === 'ip_adapter_layer') {
              // Must have model
              if (!l.ipAdapter.model) {
                problems.push(i18n.t('parameters.invoke.layer.ipAdapterNoModelSelected'));
              }
              // Model base must match
              if (l.ipAdapter.model?.base !== model?.base) {
                problems.push(i18n.t('parameters.invoke.layer.ipAdapterIncompatibleBaseModel'));
              }
              // Must have an image
              if (!l.ipAdapter.image) {
                problems.push(i18n.t('parameters.invoke.layer.ipAdapterNoImageSelected'));
              }
            }

            if (l.type === 'initial_image_layer') {
              // Must have an image
              if (!l.image) {
                problems.push(i18n.t('parameters.invoke.layer.initialImageNoImageSelected'));
              }
            }

            if (l.type === 'regional_guidance_layer') {
              // Must have a region
              if (l.maskObjects.length === 0) {
                problems.push(i18n.t('parameters.invoke.layer.rgNoRegion'));
              }
              // Must have at least 1 prompt or IP Adapter
              if (l.positivePrompt === null && l.negativePrompt === null && l.ipAdapters.length === 0) {
                problems.push(i18n.t('parameters.invoke.layer.rgNoPromptsOrIPAdapters'));
              }
              l.ipAdapters.forEach((ipAdapter) => {
                // Must have model
                if (!ipAdapter.model) {
                  problems.push(i18n.t('parameters.invoke.layer.ipAdapterNoModelSelected'));
                }
                // Model base must match
                if (ipAdapter.model?.base !== model?.base) {
                  problems.push(i18n.t('parameters.invoke.layer.ipAdapterIncompatibleBaseModel'));
                }
                // Must have an image
                if (!ipAdapter.image) {
                  problems.push(i18n.t('parameters.invoke.layer.ipAdapterNoImageSelected'));
                }
              });
            }

            if (problems.length) {
              const content = upperFirst(problems.join(', '));
              reasons.push({ prefix, content });
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
              reasons.push({ content: i18n.t('parameters.invoke.noModelForControlAdapter', { number: i + 1 }) });
            } else if (ca.model.base !== model?.base) {
              // This should never happen, just a sanity check
              reasons.push({
                content: i18n.t('parameters.invoke.incompatibleBaseModelForControlAdapter', { number: i + 1 }),
              });
            }

            if (
              !ca.controlImage ||
              (isControlNetOrT2IAdapter(ca) && !ca.processedControlImage && ca.processorType !== 'none')
            ) {
              reasons.push({ content: i18n.t('parameters.invoke.noControlImageForControlAdapter', { number: i + 1 }) });
            }
          });
      }
    }

    return { isReady: !reasons.length, reasons };
  }
);

export const useIsReadyToEnqueue = () => {
  const value = useAppSelector(selector);
  return value;
};
