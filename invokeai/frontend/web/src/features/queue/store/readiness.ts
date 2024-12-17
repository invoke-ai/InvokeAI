import { createSelector } from '@reduxjs/toolkit';
import type { AppConfig } from 'app/types/invokeai';
import type { ParamsState } from 'features/controlLayers/store/paramsSlice';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasState } from 'features/controlLayers/store/types';
import {
  getControlLayerWarnings,
  getGlobalReferenceImageWarnings,
  getInpaintMaskWarnings,
  getRasterLayerWarnings,
  getRegionalGuidanceWarnings,
} from 'features/controlLayers/store/validators';
import type { DynamicPromptsState } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { selectDynamicPromptsSlice } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { NodesState, Templates } from 'features/nodes/store/types';
import type { WorkflowSettingsState } from 'features/nodes/store/workflowSettingsSlice';
import { selectWorkflowSettingsSlice } from 'features/nodes/store/workflowSettingsSlice';
import { isImageFieldCollectionInputInstance, isImageFieldCollectionInputTemplate } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import type { UpscaleState } from 'features/parameters/store/upscaleSlice';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import i18n from 'i18next';
import { forEach, upperFirst } from 'lodash-es';
import { getConnectedEdges } from 'reactflow';

/**
 * This file contains selectors and utilities for determining the app is ready to enqueue generations. The handling
 * differs for each tab (canvas, upscaling, workflows).
 *
 * For example, the canvas tab needs to check the status of the canvas manager before enqueuing, while the workflows
 * tab needs to check the status of the nodes and their connections.
 */

const LAYER_TYPE_TO_TKEY = {
  reference_image: 'controlLayers.referenceImage',
  inpaint_mask: 'controlLayers.inpaintMask',
  regional_guidance: 'controlLayers.regionalGuidance',
  raster_layer: 'controlLayers.rasterLayer',
  control_layer: 'controlLayers.controlLayer',
} as const;

export type Reason = { prefix?: string; content: string };

const disconnectedReason = (t: typeof i18n.t) => ({ content: t('parameters.invoke.systemDisconnected') });

const getReasonsWhyCannotEnqueueWorkflowsTab = (arg: {
  isConnected: boolean;
  nodes: NodesState;
  workflowSettings: WorkflowSettingsState;
  templates: Templates;
}): Reason[] => {
  const { isConnected, nodes, workflowSettings, templates } = arg;
  const reasons: Reason[] = [];

  if (!isConnected) {
    reasons.push(disconnectedReason(i18n.t));
  }

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

  return reasons;
};

const getReasonsWhyCannotEnqueueUpscaleTab = (arg: {
  isConnected: boolean;
  upscale: UpscaleState;
  config: AppConfig;
  params: ParamsState;
}) => {
  const { isConnected, upscale, config, params } = arg;
  const reasons: Reason[] = [];

  if (!isConnected) {
    reasons.push(disconnectedReason(i18n.t));
  }

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

  return reasons;
};

const getReasonsWhyCannotEnqueueCanvasTab = (arg: {
  isConnected: boolean;
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
    isConnected,
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
  const reasons: Reason[] = [];

  if (!isConnected) {
    reasons.push(disconnectedReason(i18n.t));
  }

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

  const enabledControlLayers = canvas.controlLayers.entities.filter((controlLayer) => controlLayer.isEnabled);

  // FLUX only supports 1x Control LoRA at a time.
  const controlLoRACount = enabledControlLayers.filter(
    (controlLayer) => controlLayer.controlAdapter?.model?.type === 'control_lora'
  ).length;

  if (model?.base === 'flux' && controlLoRACount > 1) {
    reasons.push({ content: i18n.t('parameters.invoke.fluxModelMultipleControlLoRAs') });
  }

  enabledControlLayers.forEach((controlLayer, i) => {
    const layerLiteral = i18n.t('controlLayers.layer_one');
    const layerNumber = i + 1;
    const layerType = i18n.t(LAYER_TYPE_TO_TKEY['control_layer']);
    const prefix = `${layerLiteral} #${layerNumber} (${layerType})`;
    const problems = getControlLayerWarnings(controlLayer, model);

    if (problems.length) {
      const content = upperFirst(problems.map((p) => i18n.t(p)).join(', '));
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
      const problems = getGlobalReferenceImageWarnings(entity, model);

      if (problems.length) {
        const content = upperFirst(problems.map((p) => i18n.t(p)).join(', '));
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
      const problems = getRegionalGuidanceWarnings(entity, model);

      if (problems.length) {
        const content = upperFirst(problems.map((p) => i18n.t(p)).join(', '));
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
      const problems = getRasterLayerWarnings(entity, model);

      if (problems.length) {
        const content = upperFirst(problems.map((p) => i18n.t(p)).join(', '));
        reasons.push({ prefix, content });
      }
    });

  canvas.inpaintMasks.entities
    .filter((entity) => entity.isEnabled)
    .forEach((entity, i) => {
      const layerLiteral = i18n.t('controlLayers.layer_one');
      const layerNumber = i + 1;
      const layerType = i18n.t(LAYER_TYPE_TO_TKEY[entity.type]);
      const prefix = `${layerLiteral} #${layerNumber} (${layerType})`;
      const problems = getInpaintMaskWarnings(entity, model);

      if (problems.length) {
        const content = upperFirst(problems.map((p) => i18n.t(p)).join(', '));
        reasons.push({ prefix, content });
      }
    });

  return reasons;
};

export const buildSelectReasonsWhyCannotEnqueueCanvasTab = (arg: {
  isConnected: boolean;
  canvasIsFiltering: boolean;
  canvasIsTransforming: boolean;
  canvasIsRasterizing: boolean;
  canvasIsCompositing: boolean;
  canvasIsSelectingObject: boolean;
}) => {
  const {
    isConnected,
    canvasIsFiltering,
    canvasIsTransforming,
    canvasIsRasterizing,
    canvasIsCompositing,
    canvasIsSelectingObject,
  } = arg;

  return createSelector(
    selectCanvasSlice,
    selectParamsSlice,
    selectDynamicPromptsSlice,
    (canvas, params, dynamicPrompts) =>
      getReasonsWhyCannotEnqueueCanvasTab({
        isConnected,
        canvas,
        params,
        dynamicPrompts,
        canvasIsFiltering,
        canvasIsTransforming,
        canvasIsRasterizing,
        canvasIsCompositing,
        canvasIsSelectingObject,
      })
  );
};

export const buildSelectIsReadyToEnqueueCanvasTab = (arg: {
  isConnected: boolean;
  canvasIsFiltering: boolean;
  canvasIsTransforming: boolean;
  canvasIsRasterizing: boolean;
  canvasIsCompositing: boolean;
  canvasIsSelectingObject: boolean;
}) => {
  const {
    isConnected,
    canvasIsFiltering,
    canvasIsTransforming,
    canvasIsRasterizing,
    canvasIsCompositing,
    canvasIsSelectingObject,
  } = arg;

  return createSelector(
    selectCanvasSlice,
    selectParamsSlice,
    selectDynamicPromptsSlice,
    (canvas, params, dynamicPrompts) =>
      getReasonsWhyCannotEnqueueCanvasTab({
        isConnected,
        canvas,
        params,
        dynamicPrompts,
        canvasIsFiltering,
        canvasIsTransforming,
        canvasIsRasterizing,
        canvasIsCompositing,
        canvasIsSelectingObject,
      }).length === 0
  );
};

export const buildSelectReasonsWhyCannotEnqueueUpscaleTab = (arg: { isConnected: boolean }) => {
  const { isConnected } = arg;
  return createSelector(selectUpscaleSlice, selectConfigSlice, selectParamsSlice, (upscale, config, params) =>
    getReasonsWhyCannotEnqueueUpscaleTab({ isConnected, upscale, config, params })
  );
};

export const buildSelectIsReadyToEnqueueUpscaleTab = (arg: { isConnected: boolean }) => {
  const { isConnected } = arg;

  return createSelector(
    selectUpscaleSlice,
    selectConfigSlice,
    selectParamsSlice,
    (upscale, config, params) =>
      getReasonsWhyCannotEnqueueUpscaleTab({ isConnected, upscale, config, params }).length === 0
  );
};

export const buildSelectReasonsWhyCannotEnqueueWorkflowsTab = (arg: { isConnected: boolean; templates: Templates }) => {
  const { isConnected, templates } = arg;

  return createSelector(selectNodesSlice, selectWorkflowSettingsSlice, (nodes, workflowSettings) =>
    getReasonsWhyCannotEnqueueWorkflowsTab({
      isConnected,
      nodes,
      workflowSettings,
      templates,
    })
  );
};

export const buildSelectIsReadyToEnqueueWorkflowsTab = (arg: { isConnected: boolean; templates: Templates }) => {
  const { isConnected, templates } = arg;

  return createSelector(
    selectNodesSlice,
    selectWorkflowSettingsSlice,
    (nodes, workflowSettings) =>
      getReasonsWhyCannotEnqueueWorkflowsTab({
        isConnected,
        nodes,
        workflowSettings,
        templates,
      }).length === 0
  );
};

export const selectPromptsCount = createSelector(
  selectParamsSlice,
  selectDynamicPromptsSlice,
  (params, dynamicPrompts) => (getShouldProcessPrompt(params.positivePrompt) ? dynamicPrompts.prompts.length : 1)
);

export const selectWorkflowsBatchSize = createSelector(selectNodesSlice, ({ nodes }) =>
  // The batch size is the product of all batch nodes' collection sizes
  nodes.filter(isInvocationNode).reduce((batchSize, node) => {
    if (!isImageFieldCollectionInputInstance(node.data.inputs.images)) {
      return batchSize;
    }
    // If the batch size is not set, default to 1
    batchSize = batchSize || 1;
    // Multiply the batch size by the number of images in the batch
    batchSize = batchSize * (node.data.inputs.images.value?.length ?? 0);

    return batchSize;
  }, 0)
);
