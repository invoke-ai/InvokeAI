import { createSelector } from '@reduxjs/toolkit';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
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
import {
  isFloatFieldCollectionInputInstance,
  isFloatFieldCollectionInputTemplate,
  isImageFieldCollectionInputInstance,
  isImageFieldCollectionInputTemplate,
  isIntegerFieldCollectionInputInstance,
  isIntegerFieldCollectionInputTemplate,
  isStringFieldCollectionInputInstance,
  isStringFieldCollectionInputTemplate,
} from 'features/nodes/types/field';
import {
  resolveNumberFieldCollectionValue,
  validateImageFieldCollectionValue,
  validateNumberFieldCollectionValue,
  validateStringFieldCollectionValue,
} from 'features/nodes/types/fieldValidators';
import type { InvocationNode } from 'features/nodes/types/invocation';
import { isBatchNode, isInvocationNode } from 'features/nodes/types/invocation';
import type { UpscaleState } from 'features/parameters/store/upscaleSlice';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import i18n from 'i18next';
import { forEach, groupBy, negate, upperFirst } from 'lodash-es';
import { getConnectedEdges } from 'reactflow';
import { assert } from 'tsafe';

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
    const invocationNodes = nodes.nodes.filter(isInvocationNode);
    const batchNodes = invocationNodes.filter(isBatchNode);
    const nonBatchNodes = invocationNodes.filter(negate(isBatchNode));

    if (!nonBatchNodes.length) {
      reasons.push({ content: i18n.t('parameters.invoke.noNodesInGraph') });
    }

    for (const node of batchNodes) {
      if (nodes.edges.find((e) => e.source === node.id) === undefined) {
        reasons.push({ content: i18n.t('parameters.invoke.batchNodeNotConnected', { label: node.data.label }) });
      }
    }

    if (batchNodes.length > 1) {
      const groupedBatchNodes = groupBy(batchNodes, (node) => node.data.inputs['batch_group_id']?.value);
      for (const [batchGroupId, batchNodes] of Object.entries(groupedBatchNodes)) {
        if (batchGroupId === 'None') {
          // Ungrouped batch nodes may have differing collection sizes
          continue;
        }

        // But grouped batch nodes must have the same collection size
        const collectionSizes: number[] = [];

        for (const node of batchNodes) {
          if (node.data.type === 'image_batch') {
            assert(isImageFieldCollectionInputInstance(node.data.inputs.images));
            collectionSizes.push(node.data.inputs.images.value?.length ?? 0);
          } else if (node.data.type === 'string_batch') {
            assert(isStringFieldCollectionInputInstance(node.data.inputs.strings));
            collectionSizes.push(node.data.inputs.strings.value?.length ?? 0);
          } else if (node.data.type === 'float_batch') {
            assert(isFloatFieldCollectionInputInstance(node.data.inputs.floats));
            collectionSizes.push(node.data.inputs.floats.value?.length ?? 0);
          } else if (node.data.type === 'integer_batch') {
            assert(isIntegerFieldCollectionInputInstance(node.data.inputs.integers));
            collectionSizes.push(node.data.inputs.integers.value?.length ?? 0);
          }
        }

        if (collectionSizes.some((count) => count !== collectionSizes[0])) {
          reasons.push({
            content: i18n.t('parameters.invoke.batchNodeCollectionSizeMismatch', { batchGroupId }),
          });
        }
      }
    }

    nonBatchNodes.forEach((node) => {
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

        const prefix = `${node.data.label || nodeTemplate.title} -> ${field.label || fieldTemplate.title}`;

        if (fieldTemplate.required && field.value === undefined && !hasConnection) {
          reasons.push({ prefix, content: i18n.t('parameters.invoke.missingInputForField') });
        } else if (
          field.value &&
          isImageFieldCollectionInputInstance(field) &&
          isImageFieldCollectionInputTemplate(fieldTemplate)
        ) {
          const errors = validateImageFieldCollectionValue(field.value, fieldTemplate);
          reasons.push(...errors.map((error) => ({ prefix, content: error })));
        } else if (
          field.value &&
          isStringFieldCollectionInputInstance(field) &&
          isStringFieldCollectionInputTemplate(fieldTemplate)
        ) {
          const errors = validateStringFieldCollectionValue(field.value, fieldTemplate);
          reasons.push(...errors.map((error) => ({ prefix, content: error })));
        } else if (
          field.value &&
          isIntegerFieldCollectionInputInstance(field) &&
          isIntegerFieldCollectionInputTemplate(fieldTemplate)
        ) {
          const errors = validateNumberFieldCollectionValue(field, fieldTemplate);
          reasons.push(...errors.map((error) => ({ prefix, content: error })));
        } else if (
          field.value &&
          isFloatFieldCollectionInputInstance(field) &&
          isFloatFieldCollectionInputTemplate(fieldTemplate)
        ) {
          const errors = validateNumberFieldCollectionValue(field, fieldTemplate);
          reasons.push(...errors.map((error) => ({ prefix, content: error })));
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

const getBatchCollectionSize = (batchNode: InvocationNode) => {
  if (batchNode.data.type === 'image_batch') {
    assert(isImageFieldCollectionInputInstance(batchNode.data.inputs.images));
    return batchNode.data.inputs.images.value?.length ?? 0;
  } else if (batchNode.data.type === 'string_batch') {
    assert(isStringFieldCollectionInputInstance(batchNode.data.inputs.strings));
    return batchNode.data.inputs.strings.value?.length ?? 0;
  } else if (batchNode.data.type === 'float_batch') {
    assert(isFloatFieldCollectionInputInstance(batchNode.data.inputs.floats));
    return resolveNumberFieldCollectionValue(batchNode.data.inputs.floats)?.length ?? 0;
  } else if (batchNode.data.type === 'integer_batch') {
    assert(isIntegerFieldCollectionInputInstance(batchNode.data.inputs.integers));
    return resolveNumberFieldCollectionValue(batchNode.data.inputs.integers)?.length ?? 0;
  }
  return 0;
};

const buildSelectGroupBatchSizes = (batchGroupId: string) =>
  createMemoizedSelector(selectNodesSlice, ({ nodes }) => {
    return nodes
      .filter(isInvocationNode)
      .filter(isBatchNode)
      .filter((node) => node.data.inputs['batch_group_id']?.value === batchGroupId)
      .map(getBatchCollectionSize);
  });

const selectUngroupedBatchSizes = buildSelectGroupBatchSizes('None');
const selectGroup1BatchSizes = buildSelectGroupBatchSizes('Group 1');
const selectGroup2BatchSizes = buildSelectGroupBatchSizes('Group 2');
const selectGroup3BatchSizes = buildSelectGroupBatchSizes('Group 3');
const selectGroup4BatchSizes = buildSelectGroupBatchSizes('Group 4');
const selectGroup5BatchSizes = buildSelectGroupBatchSizes('Group 5');

export const selectWorkflowsBatchSize = createSelector(
  selectUngroupedBatchSizes,
  selectGroup1BatchSizes,
  selectGroup2BatchSizes,
  selectGroup3BatchSizes,
  selectGroup4BatchSizes,
  selectGroup5BatchSizes,
  (
    ungroupedBatchSizes,
    group1BatchSizes,
    group2BatchSizes,
    group3BatchSizes,
    group4BatchSizes,
    group5BatchSizes
  ): number | 'INVALID' | 'NO_BATCHES' => {
    // All batch nodes _must_ have a populated collection

    const allBatchSizes = [
      ...ungroupedBatchSizes,
      ...group1BatchSizes,
      ...group2BatchSizes,
      ...group3BatchSizes,
      ...group4BatchSizes,
      ...group5BatchSizes,
    ];

    // There are no batch nodes
    if (allBatchSizes.length === 0) {
      return 'NO_BATCHES';
    }

    // All batch nodes must have a populated collection
    if (allBatchSizes.some((size) => size === 0)) {
      return 'INVALID';
    }

    for (const group of [group1BatchSizes, group2BatchSizes, group3BatchSizes, group4BatchSizes, group5BatchSizes]) {
      // Ignore groups with no batch nodes
      if (group.length === 0) {
        continue;
      }
      // Grouped batch nodes must have the same collection size
      if (group.some((size) => size !== group[0])) {
        return 'INVALID';
      }
    }

    // Total batch size = product of all ungrouped batches and each grouped batch
    const totalBatchSize = [
      ...ungroupedBatchSizes,
      // In case of no batch nodes in a group, fall back to 1 for the product calculation
      group1BatchSizes[0] ?? 1,
      group2BatchSizes[0] ?? 1,
      group3BatchSizes[0] ?? 1,
      group4BatchSizes[0] ?? 1,
      group5BatchSizes[0] ?? 1,
    ].reduce((acc, size) => acc * size, 1);

    return totalBatchSize;
  }
);
