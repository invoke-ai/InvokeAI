import { useStore } from '@nanostores/react';
import { $isConnected } from 'app/hooks/useSocketIO';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { selectDynamicPromptsSlice } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import { selectWorkflowSettingsSlice } from 'features/nodes/store/workflowSettingsSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { selectUpscalelice } from 'features/parameters/store/upscaleSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { selectSystemSlice } from 'features/system/store/systemSlice';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import i18n from 'i18next';
import { forEach, upperFirst } from 'lodash-es';
import { useMemo } from 'react';
import { getConnectedEdges } from 'reactflow';

const LAYER_TYPE_TO_TKEY = {
  ip_adapter: 'controlLayers.ipAdapter',
  inpaint_mask: 'controlLayers.inpaintMask',
  regional_guidance: 'controlLayers.regionalGuidance',
  raster_layer: 'controlLayers.raster',
  control_layer: 'controlLayers.globalControlAdapter',
} as const;

const createSelector = (templates: Templates, isConnected: boolean) =>
  createMemoizedSelector(
    [
      selectSystemSlice,
      selectNodesSlice,
      selectWorkflowSettingsSlice,
      selectDynamicPromptsSlice,
      selectCanvasSlice,
      selectParamsSlice,
      selectUpscalelice,
      selectConfigSlice,
      selectActiveTab,
    ],
    (system, nodes, workflowSettings, dynamicPrompts, canvas, params, upscale, config, activeTabName) => {
      const { bbox } = canvas;
      const { model, positivePrompt } = params;

      const reasons: { prefix?: string; content: string }[] = [];

      // Cannot generate if not connected
      if (!isConnected) {
        reasons.push({ content: i18n.t('parameters.invoke.systemDisconnected') });
      }

      if (activeTabName === 'workflows') {
        if (workflowSettings.shouldValidateGraph) {
          if (!nodes.nodes.length) {
            reasons.push({ content: i18n.t('parameters.invoke.noNodesInGraph') });
          }

          nodes.nodes.forEach((node) => {
            if (!isInvocationNode(node)) {
              return;
            }

            const nodeTemplate = templates[node.data.type];

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
      } else if (activeTabName === 'upscaling') {
        if (!upscale.upscaleInitialImage) {
          reasons.push({ content: i18n.t('upscaling.missingUpscaleInitialImage') });
        } else if (config.maxUpscaleDimension) {
          const { width, height } = upscale.upscaleInitialImage;
          const { scale } = upscale;

          const maxPixels = config.maxUpscaleDimension ** 2;
          const upscaledPixels = width * scale * height * scale;

          if (upscaledPixels > maxPixels) {
            reasons.push({ content: i18n.t('upscaling.exceedsMaxSize') });
          }
        }
        if (!upscale.upscaleModel) {
          reasons.push({ content: i18n.t('upscaling.missingUpscaleModel') });
        }
        if (!upscale.tileControlnetModel) {
          reasons.push({ content: i18n.t('upscaling.missingTileControlNetModel') });
        }
      } else {
        if (dynamicPrompts.prompts.length === 0 && getShouldProcessPrompt(positivePrompt)) {
          reasons.push({ content: i18n.t('parameters.invoke.noPrompts') });
        }

        if (!model) {
          reasons.push({ content: i18n.t('parameters.invoke.noModelSelected') });
        }

        canvas.controlLayers.entities
          .filter((controlLayer) => controlLayer.isEnabled)
          .forEach((controlLayer, i) => {
            const layerLiteral = i18n.t('controlLayers.layer_one');
            const layerNumber = i + 1;
            const layerType = i18n.t(LAYER_TYPE_TO_TKEY['control_layer']);
            const prefix = `${layerLiteral} #${layerNumber} (${layerType})`;
            const problems: string[] = [];
            // Must have model
            if (!controlLayer.controlAdapter.model) {
              problems.push(i18n.t('parameters.invoke.layer.controlAdapterNoModelSelected'));
            }
            // Model base must match
            if (controlLayer.controlAdapter.model?.base !== model?.base) {
              problems.push(i18n.t('parameters.invoke.layer.controlAdapterIncompatibleBaseModel'));
            }
            // T2I Adapters require images have dimensions that are multiples of 64 (SD1.5) or 32 (SDXL)
            if (controlLayer.controlAdapter.type === 't2i_adapter') {
              const multiple = model?.base === 'sdxl' ? 32 : 64;
              if (bbox.rect.width % multiple !== 0 || bbox.rect.height % multiple !== 0) {
                problems.push(i18n.t('parameters.invoke.layer.t2iAdapterIncompatibleDimensions', { multiple }));
              }
            }

            if (problems.length) {
              const content = upperFirst(problems.join(', '));
              reasons.push({ prefix, content });
            }
          });

        canvas.ipAdapters.entities
          .filter((entity) => entity.isEnabled)
          .forEach((entity, i) => {
            const layerLiteral = i18n.t('controlLayers.layer_one');
            const layerNumber = i + 1;
            const layerType = i18n.t(LAYER_TYPE_TO_TKEY[entity.type]);
            const prefix = `${layerLiteral} #${layerNumber} (${layerType})`;
            const problems: string[] = [];

            // Must have model
            if (!entity.ipAdapter.model) {
              problems.push(i18n.t('parameters.invoke.layer.ipAdapterNoModelSelected'));
            }
            // Model base must match
            if (entity.ipAdapter.model?.base !== model?.base) {
              problems.push(i18n.t('parameters.invoke.layer.ipAdapterIncompatibleBaseModel'));
            }
            // Must have an image
            if (!entity.ipAdapter.image) {
              problems.push(i18n.t('parameters.invoke.layer.ipAdapterNoImageSelected'));
            }

            if (problems.length) {
              const content = upperFirst(problems.join(', '));
              reasons.push({ prefix, content });
            }
          });

        canvas.regions.entities
          .filter((entity) => entity.isEnabled)
          .forEach((entity, i) => {
            const layerLiteral = i18n.t('controlLayers.layer_one');
            const layerNumber = i + 1;
            const layerType = i18n.t(LAYER_TYPE_TO_TKEY[entity.type]);
            const prefix = `${layerLiteral} #${layerNumber} (${layerType})`;
            const problems: string[] = [];
            // Must have a region
            if (entity.objects.length === 0) {
              problems.push(i18n.t('parameters.invoke.layer.rgNoRegion'));
            }
            // Must have at least 1 prompt or IP Adapter
            if (entity.positivePrompt === null && entity.negativePrompt === null && entity.ipAdapters.length === 0) {
              problems.push(i18n.t('parameters.invoke.layer.rgNoPromptsOrIPAdapters'));
            }
            entity.ipAdapters.forEach((ipAdapter) => {
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

            if (problems.length) {
              const content = upperFirst(problems.join(', '));
              reasons.push({ prefix, content });
            }
          });

        canvas.rasterLayers.entities
          .filter((entity) => entity.isEnabled)
          .forEach((entity, i) => {
            const layerLiteral = i18n.t('controlLayers.layer_one');
            const layerNumber = i + 1;
            const layerType = i18n.t(LAYER_TYPE_TO_TKEY[entity.type]);
            const prefix = `${layerLiteral} #${layerNumber} (${layerType})`;
            const problems: string[] = [];

            if (problems.length) {
              const content = upperFirst(problems.join(', '));
              reasons.push({ prefix, content });
            }
          });
      }

      return { isReady: !reasons.length, reasons };
    }
  );

export const useIsReadyToEnqueue = () => {
  const templates = useStore($templates);
  const isConnected = useStore($isConnected);
  const selector = useMemo(() => createSelector(templates, isConnected), [templates, isConnected]);
  const value = useAppSelector(selector);
  return value;
};
