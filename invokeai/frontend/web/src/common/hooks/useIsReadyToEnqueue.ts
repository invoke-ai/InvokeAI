import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { $true } from 'app/store/nanostores/util';
import { useAppSelector } from 'app/store/storeHooks';
import type { AppConfig } from 'app/types/invokeai';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { ParamsState } from 'features/controlLayers/store/paramsSlice';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasState } from 'features/controlLayers/store/types';
import type { DynamicPromptsState } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { selectDynamicPromptsSlice } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { NodesState, Templates } from 'features/nodes/store/types';
import type { WorkflowSettingsState } from 'features/nodes/store/workflowSettingsSlice';
import { selectWorkflowSettingsSlice } from 'features/nodes/store/workflowSettingsSlice';
import { isImageFieldCollectionInputInstance, isImageFieldCollectionInputTemplate } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import type { UpscaleState } from 'features/parameters/store/upscaleSlice';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { selectSystemSlice } from 'features/system/store/systemSlice';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import i18n from 'i18next';
import { forEach, upperFirst } from 'lodash-es';
import { useMemo } from 'react';
import { getConnectedEdges } from 'reactflow';
import { $isConnected } from 'services/events/stores';

const LAYER_TYPE_TO_TKEY = {
  reference_image: 'controlLayers.referenceImage',
  inpaint_mask: 'controlLayers.inpaintMask',
  regional_guidance: 'controlLayers.regionalGuidance',
  raster_layer: 'controlLayers.rasterLayer',
  control_layer: 'controlLayers.controlLayer',
} as const;

type Reason = { prefix?: string; content: string };

const handleWorkflowsTab = (arg: {
  reasons: Reason[];
  nodes: NodesState;
  workflowSettings: WorkflowSettingsState;
  templates: Templates;
}) => {
  const { reasons, nodes, workflowSettings, templates } = arg;

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

        const baseTKeyOptions = {
          nodeLabel: node.data.label || nodeTemplate.title,
          fieldLabel: field.label || fieldTemplate.title,
        };

        if (fieldTemplate.required && field.value === undefined && !hasConnection) {
          reasons.push({ content: i18n.t('parameters.invoke.missingInputForField', baseTKeyOptions) });
          return;
        } else if (
          field.value &&
          isImageFieldCollectionInputInstance(field) &&
          isImageFieldCollectionInputTemplate(fieldTemplate)
        ) {
          // Image collections may have min or max items to validate
          // TODO(psyche): generalize this to other collection types
          if (fieldTemplate.minItems !== undefined && fieldTemplate.minItems > 0 && field.value.length === 0) {
            reasons.push({ content: i18n.t('parameters.invoke.collectionEmpty', baseTKeyOptions) });
            return;
          }
          if (fieldTemplate.minItems !== undefined && field.value.length < fieldTemplate.minItems) {
            reasons.push({
              content: i18n.t('parameters.invoke.collectionTooFewItems', {
                ...baseTKeyOptions,
                size: field.value.length,
                minItems: fieldTemplate.minItems,
              }),
            });
            return;
          }
          if (fieldTemplate.maxItems !== undefined && field.value.length > fieldTemplate.maxItems) {
            reasons.push({
              content: i18n.t('parameters.invoke.collectionTooManyItems', {
                ...baseTKeyOptions,
                size: field.value.length,
                maxItems: fieldTemplate.maxItems,
              }),
            });
            return;
          }
        }
      });
    });
  }
};

const handleUpscalingTab = (arg: {
  reasons: Reason[];
  upscale: UpscaleState;
  config: AppConfig;
  params: ParamsState;
}) => {
  const { reasons, upscale, config, params } = arg;

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
  const model = params.model;
  if (model && !['sd-1', 'sdxl'].includes(model.base)) {
    // When we are using an upsupported model, do not add the other warnings
    reasons.push({ content: i18n.t('upscaling.incompatibleBaseModel') });
  } else {
    // Using a compatible model, add all warnings
    if (!model) {
      reasons.push({ content: i18n.t('parameters.invoke.noModelSelected') });
    }
    if (!upscale.upscaleModel) {
      reasons.push({ content: i18n.t('upscaling.missingUpscaleModel') });
    }
    if (!upscale.tileControlnetModel) {
      reasons.push({ content: i18n.t('upscaling.missingTileControlNetModel') });
    }
  }
};

const handleCanvasTab = (arg: {
  reasons: Reason[];
  canvas: CanvasState;
  params: ParamsState;
  dynamicPrompts: DynamicPromptsState;
  canvasIsFiltering: boolean;
  canvasIsTransforming: boolean;
  canvasIsRasterizing: boolean;
  canvasIsCompositing: boolean;
  canvasIsSelectingObject: boolean;
}) => {
  const {
    reasons,
    canvas,
    params,
    dynamicPrompts,
    canvasIsFiltering,
    canvasIsTransforming,
    canvasIsRasterizing,
    canvasIsCompositing,
    canvasIsSelectingObject,
  } = arg;

  const { model, positivePrompt } = params;

  if (canvasIsFiltering) {
    reasons.push({ content: i18n.t('parameters.invoke.canvasIsFiltering') });
  }
  if (canvasIsTransforming) {
    reasons.push({ content: i18n.t('parameters.invoke.canvasIsTransforming') });
  }
  if (canvasIsRasterizing) {
    reasons.push({ content: i18n.t('parameters.invoke.canvasIsRasterizing') });
  }
  if (canvasIsCompositing) {
    reasons.push({ content: i18n.t('parameters.invoke.canvasIsCompositing') });
  }
  if (canvasIsSelectingObject) {
    reasons.push({ content: i18n.t('parameters.invoke.canvasIsSelectingObject') });
  }

  if (dynamicPrompts.prompts.length === 0 && getShouldProcessPrompt(positivePrompt)) {
    reasons.push({ content: i18n.t('parameters.invoke.noPrompts') });
  }

  if (!model) {
    reasons.push({ content: i18n.t('parameters.invoke.noModelSelected') });
  }

  if (model?.base === 'flux') {
    const { bbox } = canvas;

    if (!params.t5EncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noT5EncoderModelSelected') });
    }
    if (!params.clipEmbedModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noCLIPEmbedModelSelected') });
    }
    if (!params.fluxVAE) {
      reasons.push({ content: i18n.t('parameters.invoke.noFLUXVAEModelSelected') });
    }
    if (bbox.scaleMethod === 'none') {
      if (bbox.rect.width % 16 !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.fluxModelIncompatibleBboxWidth', { width: bbox.rect.width }),
        });
      }
      if (bbox.rect.height % 16 !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.fluxModelIncompatibleBboxHeight', { height: bbox.rect.height }),
        });
      }
    } else {
      if (bbox.scaledSize.width % 16 !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.fluxModelIncompatibleScaledBboxWidth', {
            width: bbox.scaledSize.width,
          }),
        });
      }
      if (bbox.scaledSize.height % 16 !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.fluxModelIncompatibleScaledBboxHeight', {
            height: bbox.scaledSize.height,
          }),
        });
      }
    }
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
      if (problems.length) {
        const content = upperFirst(problems.join(', '));
        reasons.push({ prefix, content });
      }
    });

  canvas.referenceImages.entities
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

  canvas.regionalGuidance.entities
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
      if (entity.positivePrompt === null && entity.negativePrompt === null && entity.referenceImages.length === 0) {
        problems.push(i18n.t('parameters.invoke.layer.rgNoPromptsOrIPAdapters'));
      }
      entity.referenceImages.forEach(({ ipAdapter }) => {
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
};

const createSelector = (arg: {
  templates: Templates;
  isConnected: boolean;
  canvasIsFiltering: boolean;
  canvasIsTransforming: boolean;
  canvasIsRasterizing: boolean;
  canvasIsCompositing: boolean;
  canvasIsSelectingObject: boolean;
}) => {
  const {
    templates,
    isConnected,
    canvasIsFiltering,
    canvasIsTransforming,
    canvasIsRasterizing,
    canvasIsCompositing,
    canvasIsSelectingObject,
  } = arg;
  return createMemoizedSelector(
    [
      selectSystemSlice,
      selectNodesSlice,
      selectWorkflowSettingsSlice,
      selectDynamicPromptsSlice,
      selectCanvasSlice,
      selectParamsSlice,
      selectUpscaleSlice,
      selectConfigSlice,
      selectActiveTab,
    ],
    (system, nodes, workflowSettings, dynamicPrompts, canvas, params, upscale, config, activeTabName) => {
      const reasons: Reason[] = [];

      // Cannot generate if not connected
      if (!isConnected) {
        reasons.push({ content: i18n.t('parameters.invoke.systemDisconnected') });
      }

      if (activeTabName === 'workflows') {
        handleWorkflowsTab({ reasons, nodes, workflowSettings, templates });
      } else if (activeTabName === 'upscaling') {
        handleUpscalingTab({ reasons, upscale, config, params });
      } else {
        handleCanvasTab({
          reasons,
          canvas,
          params,
          dynamicPrompts,
          canvasIsFiltering,
          canvasIsTransforming,
          canvasIsRasterizing,
          canvasIsCompositing,
          canvasIsSelectingObject,
        });
      }

      return { isReady: !reasons.length, reasons };
    }
  );
};

export const useIsReadyToEnqueue = () => {
  const templates = useStore($templates);
  const isConnected = useStore($isConnected);
  const canvasManager = useCanvasManagerSafe();
  const canvasIsFiltering = useStore(canvasManager?.stateApi.$isFiltering ?? $true);
  const canvasIsTransforming = useStore(canvasManager?.stateApi.$isTransforming ?? $true);
  const canvasIsRasterizing = useStore(canvasManager?.stateApi.$isRasterizing ?? $true);
  const canvasIsSelectingObject = useStore(canvasManager?.stateApi.$isSegmenting ?? $true);
  const canvasIsCompositing = useStore(canvasManager?.compositor.$isBusy ?? $true);
  const selector = useMemo(
    () =>
      createSelector({
        templates,
        isConnected,
        canvasIsFiltering,
        canvasIsTransforming,
        canvasIsRasterizing,
        canvasIsCompositing,
        canvasIsSelectingObject,
      }),
    [
      templates,
      isConnected,
      canvasIsFiltering,
      canvasIsTransforming,
      canvasIsRasterizing,
      canvasIsCompositing,
      canvasIsSelectingObject,
    ]
  );
  const value = useAppSelector(selector);
  return value;
};
