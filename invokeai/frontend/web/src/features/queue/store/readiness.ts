import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import { $false } from 'app/store/nanostores/util';
import type { AppDispatch, AppStore } from 'app/store/store';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { debounce, groupBy, upperFirst } from 'es-toolkit/compat';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectAddedLoRAs } from 'features/controlLayers/store/lorasSlice';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasState, LoRA, ParamsState, RefImagesState } from 'features/controlLayers/store/types';
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
import { SUPPORTS_REF_IMAGES_BASE_MODELS } from 'features/modelManagerV2/models';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { NodesState, Templates } from 'features/nodes/store/types';
import { getInvocationNodeErrors } from 'features/nodes/store/util/fieldValidators';
import type { WorkflowSettingsState } from 'features/nodes/store/workflowSettingsSlice';
import { selectWorkflowSettingsSlice } from 'features/nodes/store/workflowSettingsSlice';
import { isBatchNode, isExecutableNode, isInvocationNode } from 'features/nodes/types/invocation';
import { resolveBatchValue } from 'features/nodes/util/node/resolveBatchValue';
import type { UpscaleState } from 'features/parameters/store/upscaleSlice';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { getGridSize } from 'features/parameters/util/optimalDimension';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { TabName } from 'features/ui/store/uiTypes';
import i18n from 'i18next';
import { atom, computed } from 'nanostores';
import { useEffect } from 'react';
import type { MainModelConfig } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

/**
 * This file contains selectors and utilities for determining the app is ready to enqueue generations. The handling
 * differs for each tab (canvas, upscaling, workflows).
 *
 * For example, the canvas tab needs to check the status of the canvas manager before enqueuing, while the workflows
 * tab needs to check the status of the nodes and their connections.
 *
 * A global store that contains the reasons why the app is not ready to enqueue generations. State changes are debounced
 * to reduce the number of times we run the fairly involved readiness checks.
 */

const LAYER_TYPE_TO_TKEY = {
  reference_image: 'controlLayers.referenceImage',
  inpaint_mask: 'controlLayers.inpaintMask',
  regional_guidance: 'controlLayers.regionalGuidance',
  raster_layer: 'controlLayers.rasterLayer',
  control_layer: 'controlLayers.controlLayer',
} as const;

export type Reason = { prefix?: string; content: string };

export const $reasonsWhyCannotEnqueue = atom<Reason[]>([]);
export const $isReadyToEnqueue = computed($reasonsWhyCannotEnqueue, (reasons) => reasons.length === 0);

type UpdateReasonsArg = {
  tab: TabName;
  isConnected: boolean;
  canvas: CanvasState;
  params: ParamsState;
  refImages: RefImagesState;
  dynamicPrompts: DynamicPromptsState;
  canvasIsFiltering: boolean;
  canvasIsTransforming: boolean;
  canvasIsRasterizing: boolean;
  canvasIsCompositing: boolean;
  canvasIsSelectingObject: boolean;
  nodes: NodesState;
  workflowSettings: WorkflowSettingsState;
  templates: Templates;
  upscale: UpscaleState;
  loras: LoRA[];
  store: AppStore;
};

const debouncedUpdateReasons = debounce(async (arg: UpdateReasonsArg) => {
  const {
    tab,
    isConnected,
    canvas,
    params,
    refImages,
    dynamicPrompts,
    canvasIsFiltering,
    canvasIsTransforming,
    canvasIsRasterizing,
    canvasIsCompositing,
    canvasIsSelectingObject,
    nodes,
    workflowSettings,
    templates,
    upscale,
    loras,
    store,
  } = arg;
  if (tab === 'generate') {
    const model = selectMainModelConfig(store.getState());
    const reasons = await getReasonsWhyCannotEnqueueGenerateTab({
      isConnected,
      model,
      params,
      refImages,
      dynamicPrompts,
      loras,
    });
    $reasonsWhyCannotEnqueue.set(reasons);
  } else if (tab === 'canvas') {
    const model = selectMainModelConfig(store.getState());
    const reasons = await getReasonsWhyCannotEnqueueCanvasTab({
      isConnected,
      model,
      canvas,
      params,
      refImages,
      dynamicPrompts,
      canvasIsFiltering,
      canvasIsTransforming,
      canvasIsRasterizing,
      canvasIsCompositing,
      canvasIsSelectingObject,
      loras,
    });
    $reasonsWhyCannotEnqueue.set(reasons);
  } else if (tab === 'workflows') {
    const reasons = await getReasonsWhyCannotEnqueueWorkflowsTab({
      dispatch: store.dispatch,
      nodesState: nodes,
      workflowSettingsState: workflowSettings,
      isConnected,
      templates,
    });
    $reasonsWhyCannotEnqueue.set(reasons);
  } else if (tab === 'upscaling') {
    const reasons = getReasonsWhyCannotEnqueueUpscaleTab({
      isConnected,
      upscale,
      params,
      loras,
    });
    $reasonsWhyCannotEnqueue.set(reasons);
  } else {
    $reasonsWhyCannotEnqueue.set(EMPTY_ARRAY);
  }
}, 300);

export const useReadinessWatcher = () => {
  useAssertSingleton('useReadinessWatcher');
  const store = useAppStore();
  const canvasManager = useCanvasManagerSafe();
  const tab = useAppSelector(selectActiveTab);
  const canvas = useAppSelector(selectCanvasSlice);
  const params = useAppSelector(selectParamsSlice);
  const refImages = useAppSelector(selectRefImagesSlice);
  const dynamicPrompts = useAppSelector(selectDynamicPromptsSlice);
  const nodes = useAppSelector(selectNodesSlice);
  const workflowSettings = useAppSelector(selectWorkflowSettingsSlice);
  const upscale = useAppSelector(selectUpscaleSlice);
  const loras = useAppSelector(selectAddedLoRAs);
  const templates = useStore($templates);
  const isConnected = useStore($isConnected);
  const canvasIsFiltering = useStore(canvasManager?.stateApi.$isFiltering ?? $false);
  const canvasIsTransforming = useStore(canvasManager?.stateApi.$isTransforming ?? $false);
  const canvasIsRasterizing = useStore(canvasManager?.stateApi.$isRasterizing ?? $false);
  const canvasIsSelectingObject = useStore(canvasManager?.stateApi.$isSegmenting ?? $false);
  const canvasIsCompositing = useStore(canvasManager?.compositor.$isBusy ?? $false);
  useEffect(() => {
    debouncedUpdateReasons({
      tab,
      isConnected,
      canvas,
      params,
      refImages,
      dynamicPrompts,
      canvasIsFiltering,
      canvasIsTransforming,
      canvasIsRasterizing,
      canvasIsCompositing,
      canvasIsSelectingObject,
      nodes,
      workflowSettings,
      templates,
      upscale,
      loras,
      store,
    });
  }, [
    store,
    canvas,
    refImages,
    canvasIsCompositing,
    canvasIsFiltering,
    canvasIsRasterizing,
    canvasIsSelectingObject,
    canvasIsTransforming,
    dynamicPrompts,
    isConnected,
    nodes,
    params,
    tab,
    templates,
    upscale,
    workflowSettings,
    loras,
  ]);
};

const disconnectedReason = (t: typeof i18n.t) => ({ content: t('parameters.invoke.systemDisconnected') });

const getReasonsWhyCannotEnqueueGenerateTab = (arg: {
  isConnected: boolean;
  model: MainModelConfig | null | undefined;
  params: ParamsState;
  refImages: RefImagesState;
  loras: LoRA[];
  dynamicPrompts: DynamicPromptsState;
}) => {
  const { isConnected, model, params, refImages, loras, dynamicPrompts } = arg;
  const { positivePrompt } = params;
  const reasons: Reason[] = [];

  if (!isConnected) {
    reasons.push(disconnectedReason(i18n.t));
  }

  if (dynamicPrompts.prompts.length === 0 && getShouldProcessPrompt(positivePrompt)) {
    reasons.push({ content: i18n.t('parameters.invoke.noPrompts') });
  }

  if (!model) {
    reasons.push({ content: i18n.t('parameters.invoke.noModelSelected') });
  }

  if (model?.base === 'flux') {
    if (!params.t5EncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noT5EncoderModelSelected') });
    }
    if (!params.clipEmbedModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noCLIPEmbedModelSelected') });
    }
    if (!params.fluxVAE) {
      reasons.push({ content: i18n.t('parameters.invoke.noFLUXVAEModelSelected') });
    }
  }

  // FLUX.2 (Klein) extracts Qwen3 encoder and VAE from main model - no separate selections needed

  if (model?.base === 'z-image') {
    // Check if VAE source is available (either separate VAE or Qwen3 Source)
    const hasVaeSource = params.zImageVaeModel !== null || params.zImageQwen3SourceModel !== null;
    if (!hasVaeSource) {
      reasons.push({ content: i18n.t('parameters.invoke.noZImageVaeSourceSelected') });
    }
    // Check if Qwen3 Encoder source is available (either separate Encoder or Qwen3 Source)
    const hasQwen3Source = params.zImageQwen3EncoderModel !== null || params.zImageQwen3SourceModel !== null;
    if (!hasQwen3Source) {
      reasons.push({ content: i18n.t('parameters.invoke.noZImageQwen3EncoderSourceSelected') });
    }
  }

  if (model) {
    for (const lora of loras.filter(({ isEnabled }) => isEnabled === true)) {
      if (model.base !== lora.model.base) {
        reasons.push({ content: i18n.t('parameters.invoke.incompatibleLoRAs') });
        // Just add the warning once.
        break;
      }
    }
  }

  if (model && SUPPORTS_REF_IMAGES_BASE_MODELS.includes(model.base)) {
    const enabledRefImages = refImages.entities.filter(({ isEnabled }) => isEnabled);

    enabledRefImages.forEach((entity, i) => {
      const layerNumber = i + 1;
      const refImageLiteral = i18n.t(LAYER_TYPE_TO_TKEY['reference_image']);
      const prefix = `${refImageLiteral} #${layerNumber}`;
      const problems = getGlobalReferenceImageWarnings(entity, model);

      if (problems.length) {
        const content = upperFirst(problems.map((p) => i18n.t(p)).join(', '));
        reasons.push({ prefix, content });
      }
    });
  }

  return reasons;
};
const getReasonsWhyCannotEnqueueWorkflowsTab = async (arg: {
  dispatch: AppDispatch;
  nodesState: NodesState;
  workflowSettingsState: WorkflowSettingsState;
  isConnected: boolean;
  templates: Templates;
}): Promise<Reason[]> => {
  const { dispatch, nodesState, workflowSettingsState, isConnected, templates } = arg;
  const reasons: Reason[] = [];

  if (!isConnected) {
    reasons.push(disconnectedReason(i18n.t));
  }

  if (workflowSettingsState.shouldValidateGraph) {
    const { nodes, edges } = nodesState;
    const invocationNodes = nodes.filter(isInvocationNode);
    const batchNodes = invocationNodes.filter(isBatchNode);
    const executableNodes = invocationNodes.filter(isExecutableNode);

    if (!executableNodes.length) {
      reasons.push({ content: i18n.t('parameters.invoke.noNodesInGraph') });
    }

    for (const node of batchNodes) {
      if (edges.find((e) => e.source === node.id) === undefined) {
        reasons.push({ content: i18n.t('parameters.invoke.batchNodeNotConnected', { label: node.data.label }) });
      }
    }

    if (batchNodes.length > 0) {
      const batchSizes: number[] = [];
      const groupedBatchNodes = groupBy(batchNodes, (node) => node.data.inputs['batch_group_id']?.value);
      for (const [batchGroupId, batchNodes] of Object.entries(groupedBatchNodes)) {
        // But grouped batch nodes must have the same collection size
        const groupBatchSizes: number[] = [];

        for (const node of batchNodes) {
          const size = (await resolveBatchValue({ dispatch, nodesState, node })).length;
          if (batchGroupId === 'None') {
            // Ungrouped batch nodes may have differing collection sizes
            batchSizes.push(size);
          } else {
            groupBatchSizes.push(size);
          }
        }

        if (groupBatchSizes.some((count) => count !== groupBatchSizes[0])) {
          reasons.push({
            content: i18n.t('parameters.invoke.batchNodeCollectionSizeMismatch', { batchGroupId }),
          });
        }

        if (groupBatchSizes[0] !== undefined) {
          batchSizes.push(groupBatchSizes[0]);
        }
      }

      if (batchSizes.some((size) => size === 0)) {
        reasons.push({ content: i18n.t('parameters.invoke.batchNodeEmptyCollection') });
      }
    }

    invocationNodes.forEach((node) => {
      if (!isInvocationNode(node)) {
        return;
      }

      const errors = getInvocationNodeErrors(node.data.id, templates, nodesState);

      for (const error of errors) {
        if (error.type === 'node-error') {
          reasons.push({ content: error.issue });
        } else {
          // error.type === 'field-error'
          reasons.push({ prefix: error.prefix, content: error.issue });
        }
      }
    });
  }

  return reasons;
};

const getReasonsWhyCannotEnqueueUpscaleTab = (arg: {
  isConnected: boolean;
  upscale: UpscaleState;
  params: ParamsState;
  loras: LoRA[];
}) => {
  const { isConnected, upscale, params, loras } = arg;
  const reasons: Reason[] = [];

  if (!isConnected) {
    reasons.push(disconnectedReason(i18n.t));
  }

  if (!upscale.upscaleInitialImage) {
    reasons.push({ content: i18n.t('upscaling.missingUpscaleInitialImage') });
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
    if (model) {
      for (const lora of loras.filter(({ isEnabled }) => isEnabled === true)) {
        if (model.base !== lora.model.base) {
          reasons.push({ content: i18n.t('parameters.invoke.incompatibleLoRAs') });
          // Just add the warning once.
          break;
        }
      }
    }
  }

  return reasons;
};

const getReasonsWhyCannotEnqueueCanvasTab = (arg: {
  isConnected: boolean;
  model: MainModelConfig | null | undefined;
  canvas: CanvasState;
  params: ParamsState;
  refImages: RefImagesState;
  loras: LoRA[];
  dynamicPrompts: DynamicPromptsState;
  canvasIsFiltering: boolean;
  canvasIsTransforming: boolean;
  canvasIsRasterizing: boolean;
  canvasIsCompositing: boolean;
  canvasIsSelectingObject: boolean;
}) => {
  const {
    isConnected,
    model,
    canvas,
    params,
    refImages,
    loras,
    dynamicPrompts,
    canvasIsFiltering,
    canvasIsTransforming,
    canvasIsRasterizing,
    canvasIsCompositing,
    canvasIsSelectingObject,
  } = arg;
  const { positivePrompt } = params;
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
    if (!params.t5EncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noT5EncoderModelSelected') });
    }
    if (!params.clipEmbedModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noCLIPEmbedModelSelected') });
    }
    if (!params.fluxVAE) {
      reasons.push({ content: i18n.t('parameters.invoke.noFLUXVAEModelSelected') });
    }

    const { bbox } = canvas;
    const gridSize = getGridSize('flux');

    if (bbox.scaleMethod === 'none') {
      if (bbox.rect.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxWidth', {
            model: 'FLUX',
            width: bbox.rect.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.rect.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxHeight', {
            model: 'FLUX',
            height: bbox.rect.height,
            multiple: gridSize,
          }),
        });
      }
    } else {
      if (bbox.scaledSize.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxWidth', {
            model: 'FLUX',
            width: bbox.scaledSize.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.scaledSize.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxHeight', {
            model: 'FLUX',
            height: bbox.scaledSize.height,
            multiple: gridSize,
          }),
        });
      }
    }
  }

  if (model?.base === 'flux2') {
    // FLUX.2 (Klein) extracts Qwen3 encoder and VAE from main model - no separate selections needed

    const { bbox } = canvas;
    const gridSize = getGridSize('flux'); // FLUX.2 uses same grid size as FLUX.1

    if (bbox.scaleMethod === 'none') {
      if (bbox.rect.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxWidth', {
            model: 'FLUX.2',
            width: bbox.rect.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.rect.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxHeight', {
            model: 'FLUX.2',
            height: bbox.rect.height,
            multiple: gridSize,
          }),
        });
      }
    } else {
      if (bbox.scaledSize.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxWidth', {
            model: 'FLUX.2',
            width: bbox.scaledSize.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.scaledSize.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxHeight', {
            model: 'FLUX.2',
            height: bbox.scaledSize.height,
            multiple: gridSize,
          }),
        });
      }
    }
  }

  if (model?.base === 'cogview4') {
    const { bbox } = canvas;
    const gridSize = getGridSize('cogview4');

    if (bbox.scaleMethod === 'none') {
      if (bbox.rect.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxWidth', {
            model: 'CogView4',
            width: bbox.rect.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.rect.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxHeight', {
            model: 'CogView4',
            height: bbox.rect.height,
            multiple: gridSize,
          }),
        });
      }
    } else {
      if (bbox.scaledSize.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxWidth', {
            model: 'CogView4',
            width: bbox.scaledSize.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.scaledSize.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxHeight', {
            model: 'CogView4',
            height: bbox.scaledSize.height,
            multiple: gridSize,
          }),
        });
      }
    }
  }

  if (model?.base === 'z-image') {
    // Check if VAE source is available (either separate VAE or Qwen3 Source)
    const hasVaeSource = params.zImageVaeModel !== null || params.zImageQwen3SourceModel !== null;
    if (!hasVaeSource) {
      reasons.push({ content: i18n.t('parameters.invoke.noZImageVaeSourceSelected') });
    }
    // Check if Qwen3 Encoder source is available (either separate Encoder or Qwen3 Source)
    const hasQwen3Source = params.zImageQwen3EncoderModel !== null || params.zImageQwen3SourceModel !== null;
    if (!hasQwen3Source) {
      reasons.push({ content: i18n.t('parameters.invoke.noZImageQwen3EncoderSourceSelected') });
    }
  }

  if (model) {
    for (const lora of loras.filter(({ isEnabled }) => isEnabled === true)) {
      if (model.base !== lora.model.base) {
        reasons.push({ content: i18n.t('parameters.invoke.incompatibleLoRAs') });
        // Just add the warning once.
        break;
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

  if (model && SUPPORTS_REF_IMAGES_BASE_MODELS.includes(model.base)) {
    const enabledRefImages = refImages.entities.filter(({ isEnabled }) => isEnabled);

    enabledRefImages.forEach((entity, i) => {
      const layerNumber = i + 1;
      const refImageLiteral = i18n.t(LAYER_TYPE_TO_TKEY['reference_image']);
      const prefix = `${refImageLiteral} #${layerNumber}`;
      const problems = getGlobalReferenceImageWarnings(entity, model);

      if (problems.length) {
        const content = upperFirst(problems.map((p) => i18n.t(p)).join(', '));
        reasons.push({ prefix, content });
      }
    });
  }

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

export const selectPromptsCount = createSelector(
  selectParamsSlice,
  selectDynamicPromptsSlice,
  (params, dynamicPrompts) => (getShouldProcessPrompt(params.positivePrompt) ? dynamicPrompts.prompts.length : 1)
);
