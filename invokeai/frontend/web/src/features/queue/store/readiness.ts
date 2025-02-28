import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppStore } from 'app/store/nanostores/store';
import { $true } from 'app/store/nanostores/util';
import type { AppDispatch, AppStore } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import type { AppConfig } from 'app/types/invokeai';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
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
import { selectConfigSlice } from 'features/system/store/configSlice';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { TabName } from 'features/ui/store/uiTypes';
import i18n from 'i18next';
import { debounce, groupBy, upperFirst } from 'lodash-es';
import { atom, computed } from 'nanostores';
import { useEffect } from 'react';
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

const debouncedUpdateReasons = debounce(
  async (
    tab: TabName,
    isConnected: boolean,
    canvas: CanvasState,
    params: ParamsState,
    dynamicPrompts: DynamicPromptsState,
    canvasIsFiltering: boolean,
    canvasIsTransforming: boolean,
    canvasIsRasterizing: boolean,
    canvasIsCompositing: boolean,
    canvasIsSelectingObject: boolean,
    nodes: NodesState,
    workflowSettings: WorkflowSettingsState,
    templates: Templates,
    upscale: UpscaleState,
    config: AppConfig,
    store: AppStore
  ) => {
    if (tab === 'canvas') {
      const reasons = await getReasonsWhyCannotEnqueueCanvasTab({
        isConnected,
        canvas,
        params,
        dynamicPrompts,
        canvasIsFiltering,
        canvasIsTransforming,
        canvasIsRasterizing,
        canvasIsCompositing,
        canvasIsSelectingObject,
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
        config,
        params,
      });
      $reasonsWhyCannotEnqueue.set(reasons);
    } else {
      $reasonsWhyCannotEnqueue.set(EMPTY_ARRAY);
    }
  },
  300
);

export const useReadinessWatcher = () => {
  useAssertSingleton('useReadinessWatcher');
  const store = useAppStore();
  const canvasManager = useCanvasManagerSafe();
  const tab = useAppSelector(selectActiveTab);
  const canvas = useAppSelector(selectCanvasSlice);
  const params = useAppSelector(selectParamsSlice);
  const dynamicPrompts = useAppSelector(selectDynamicPromptsSlice);
  const nodes = useAppSelector(selectNodesSlice);
  const workflowSettings = useAppSelector(selectWorkflowSettingsSlice);
  const upscale = useAppSelector(selectUpscaleSlice);
  const config = useAppSelector(selectConfigSlice);
  const templates = useStore($templates);
  const isConnected = useStore($isConnected);
  const canvasIsFiltering = useStore(canvasManager?.stateApi.$isFiltering ?? $true);
  const canvasIsTransforming = useStore(canvasManager?.stateApi.$isTransforming ?? $true);
  const canvasIsRasterizing = useStore(canvasManager?.stateApi.$isRasterizing ?? $true);
  const canvasIsSelectingObject = useStore(canvasManager?.stateApi.$isSegmenting ?? $true);
  const canvasIsCompositing = useStore(canvasManager?.compositor.$isBusy ?? $true);

  useEffect(() => {
    debouncedUpdateReasons(
      tab,
      isConnected,
      canvas,
      params,
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
      config,
      store
    );
  }, [
    store,
    canvas,
    canvasIsCompositing,
    canvasIsFiltering,
    canvasIsRasterizing,
    canvasIsSelectingObject,
    canvasIsTransforming,
    config,
    dynamicPrompts,
    isConnected,
    nodes,
    params,
    tab,
    templates,
    upscale,
    workflowSettings,
  ]);
};

const disconnectedReason = (t: typeof i18n.t) => ({ content: t('parameters.invoke.systemDisconnected') });

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

export const selectPromptsCount = createSelector(
  selectParamsSlice,
  selectDynamicPromptsSlice,
  (params, dynamicPrompts) => (getShouldProcessPrompt(params.positivePrompt) ? dynamicPrompts.prompts.length : 1)
);
