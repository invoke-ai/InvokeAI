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
import {
  isValidKrea2RebalanceWeights,
  selectMainModelConfig,
  selectParamsSlice,
} from 'features/controlLayers/store/paramsSlice';
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
import { isFlux2KleinQwen3Compatible } from 'features/parameters/util/flux2Klein';
import { getGridSize, getPidScale } from 'features/parameters/util/optimalDimension';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { TabName } from 'features/ui/store/uiTypes';
import i18n from 'i18next';
import { atom, computed } from 'nanostores';
import { useEffect } from 'react';
import { selectFlux2DevDiffusersModels, selectFlux2DiffusersModels } from 'services/api/hooks/modelsByType';
import type { MainOrExternalModelConfig } from 'services/api/types';
import {
  isExternalApiModelConfig,
  isSelfContainedSDNQFlux1Pipeline,
  isSelfContainedSDNQPipeline,
  isWanSingleFileMainModelConfig,
} from 'services/api/types';
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
    const flux2DiffusersModels = selectFlux2DiffusersModels(store.getState());
    const hasFlux2DiffusersVaeSource = flux2DiffusersModels.length > 0;
    const modelVariant = model && 'variant' in model ? model.variant : undefined;
    const hasFlux2DiffusersQwen3Source = flux2DiffusersModels.some(
      (m) => 'variant' in m && isFlux2KleinQwen3Compatible(m.variant, modelVariant)
    );
    const hasFlux2DevDiffusersSource = selectFlux2DevDiffusersModels(store.getState()).length > 0;
    const reasons = await getReasonsWhyCannotEnqueueGenerateTab({
      isConnected,
      model,
      params,
      refImages,
      dynamicPrompts,
      loras,
      hasFlux2DiffusersVaeSource,
      hasFlux2DiffusersQwen3Source,
      hasFlux2DevDiffusersSource,
    });
    $reasonsWhyCannotEnqueue.set(reasons);
  } else if (tab === 'canvas') {
    const model = selectMainModelConfig(store.getState());
    const flux2DiffusersModels = selectFlux2DiffusersModels(store.getState());
    const hasFlux2DiffusersVaeSource = flux2DiffusersModels.length > 0;
    const modelVariant = model && 'variant' in model ? model.variant : undefined;
    const hasFlux2DiffusersQwen3Source = flux2DiffusersModels.some(
      (m) => 'variant' in m && isFlux2KleinQwen3Compatible(m.variant, modelVariant)
    );
    const hasFlux2DevDiffusersSource = selectFlux2DevDiffusersModels(store.getState()).length > 0;
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
      hasFlux2DiffusersVaeSource,
      hasFlux2DiffusersQwen3Source,
      hasFlux2DevDiffusersSource,
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

/** Pre-flight for single-file Wan mains, shared by the generate and canvas tabs so the
 *  two can't drift. Mirrors what `WanModelLoaderInvocation` actually enforces.
 *
 *  Keep in step with the auto-fill in `modelSelected.ts`: if that doesn't offer to
 *  populate the slots this demands, selecting the model just blocks Invoke with
 *  nothing the user can act on.
 *
 *  Note there is deliberately no check on the A14B expert pairing. Since #9505 the
 *  loader takes the pairing from the wiring rather than the filename tag, so an unpaired
 *  or untagged A14B runs with a warning instead of raising — the only hard error left is
 *  two files claiming the *same* expert, which the pickers already prevent by offering
 *  each slot a different list. Blocking here on `expert !== 'high'` would stop a
 *  generation the backend is happy to run. */
const pushWanSingleFileReasons = (params: ParamsState, reasons: Reason[]): void => {
  // Single-file Wan mains (GGUF or safetensors checkpoint) carry only the transformer;
  // VAE + UMT5-XXL encoder must come from standalone models or the Component Source.
  const hasVaeSource = params.wanVaeModel !== null || params.wanComponentSource !== null;
  const hasEncoderSource = params.wanT5EncoderModel !== null || params.wanComponentSource !== null;
  if (!hasVaeSource || !hasEncoderSource) {
    reasons.push({ content: i18n.t('parameters.invoke.noWanComponentSourceSelected') });
  }
};

export const getReasonsWhyCannotEnqueueGenerateTab = (arg: {
  isConnected: boolean;
  model: MainOrExternalModelConfig | null | undefined;
  params: ParamsState;
  refImages: RefImagesState;
  loras: LoRA[];
  dynamicPrompts: DynamicPromptsState;
  hasFlux2DiffusersVaeSource: boolean;
  hasFlux2DiffusersQwen3Source: boolean;
  hasFlux2DevDiffusersSource: boolean;
}) => {
  const {
    isConnected,
    model,
    params,
    refImages,
    loras,
    dynamicPrompts,
    hasFlux2DiffusersVaeSource,
    hasFlux2DiffusersQwen3Source,
    hasFlux2DevDiffusersSource,
  } = arg;
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

  if (!model) {
    // nothing else to validate
  } else if (isExternalApiModelConfig(model)) {
    // external models don't require local sub-models
  } else if (model.base === 'flux') {
    // A complete SDNQ FLUX.1 pipeline install ships its own T5, CLIP and VAE, and the model loader
    // node falls back to them, so requiring the standalone selections here would keep that path
    // unreachable from the UI. Anything else (single-file, GGUF, BnB) still needs all three.
    const mainSuppliesComponents = isSelfContainedSDNQFlux1Pipeline(model);
    if (!mainSuppliesComponents) {
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
    if (params.pidMode !== 'off') {
      if (!params.pidDecoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
      }
      if (!params.gemma2EncoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
      }
    }
  }

  if (model?.base === 'flux2') {
    // A FLUX.2 model is a self-sufficient source when its config exposes the diffusers-style
    // submodels (transformer/vae/text_encoder/tokenizer). Plain Diffusers pipelines always do; an
    // SDNQ pipeline qualifies only when it ships all of them — a truthy submodels dict is not enough,
    // since a partial pipeline may expose only the transformer and the backend would then request
    // missing fixed subfolders. Single-file / GGUF models have no submodels and need a standalone
    // VAE + text encoder, or a Diffusers source of the matching variant family.
    const mainIsPipeline =
      model.format === 'diffusers' ||
      ((model as { format?: unknown }).format === 'sdnq_quantized' && isSelfContainedSDNQPipeline(model));
    if (!mainIsPipeline) {
      if ('variant' in model && model.variant === 'dev') {
        // FLUX.2 [dev]: needs FLUX.2 VAE + Mistral text encoder.
        if (!params.flux2VaeModel && !hasFlux2DevDiffusersSource) {
          reasons.push({ content: i18n.t('parameters.invoke.noFlux2DevVaeModelSelected') });
        }
        if (!params.flux2DevMistralEncoderModel && !hasFlux2DevDiffusersSource) {
          reasons.push({ content: i18n.t('parameters.invoke.noFlux2DevMistralEncoderModelSelected') });
        }
      } else {
        // FLUX.2 Klein: needs FLUX.2 VAE + Qwen3 text encoder (variant-matched).
        if (!params.flux2VaeModel && !hasFlux2DiffusersVaeSource) {
          reasons.push({ content: i18n.t('parameters.invoke.noFlux2KleinVaeModelSelected') });
        }
        if (!params.kleinQwen3EncoderModel && !hasFlux2DiffusersQwen3Source) {
          reasons.push({ content: i18n.t('parameters.invoke.noFlux2KleinQwen3EncoderModelSelected') });
        }
      }
    }
  }

  if (model?.base === 'flux2' && params.pidMode !== 'off') {
    // PiD decode (any FLUX.2 format) needs both a PiD decoder and the Gemma-2 caption encoder.
    if (!params.pidDecoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
    }
    if (!params.gemma2EncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
    }
  }

  if (model?.base === 'sd-3' && params.pidMode !== 'off') {
    // PiD decode needs both a PiD decoder and the Gemma-2 caption encoder.
    if (!params.pidDecoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
    }
    if (!params.gemma2EncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
    }
  }

  if (model?.base === 'sdxl' && params.pidMode !== 'off') {
    // PiD decode needs the decoder + Gemma-2 encoder, and is not compatible with the SDXL Refiner.
    if (!params.pidDecoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
    }
    if (!params.gemma2EncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
    }
    if (params.refinerModel) {
      reasons.push({ content: i18n.t('parameters.invoke.pidIncompatibleWithRefiner') });
    }
  }

  if (model?.base === 'qwen-image' && model.format === 'gguf_quantized') {
    // GGUF needs sources for VAE + encoder. Each can come from either a standalone
    // model or the Component Source (Diffusers).
    const hasVaeSource = params.qwenImageVaeModel !== null || params.qwenImageComponentSource !== null;
    const hasEncoderSource = params.qwenImageQwenVLEncoderModel !== null || params.qwenImageComponentSource !== null;
    if (!hasVaeSource || !hasEncoderSource) {
      reasons.push({ content: i18n.t('parameters.invoke.noQwenImageComponentSourceSelected') });
    }
  }

  if (model?.base === 'qwen-image' && params.pidMode !== 'off') {
    // PiD decode (any Qwen-Image format) needs both a PiD decoder and the Gemma-2 caption encoder.
    if (!params.pidDecoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
    }
    if (!params.gemma2EncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
    }
  }

  if (model && isWanSingleFileMainModelConfig(model)) {
    pushWanSingleFileReasons(params, reasons);
  }

  if (model?.base === 'z-image') {
    // An SDNQ-quantized Z-Image pipeline install is self-contained: it ships the VAE and Qwen3
    // encoder (text_encoder + tokenizer) as submodels of the main model, so no separate component
    // source is required. A truthy submodels dict is not enough — a partial pipeline may expose only
    // some submodels — so require every one the loader needs. Single-file / GGUF Z-Image models
    // don't have submodels and still need a standalone VAE + Qwen3 (or a Qwen3 Source model).
    const mainIsSelfContainedPipeline =
      (model as { format?: unknown }).format === 'sdnq_quantized' && isSelfContainedSDNQPipeline(model);
    if (!mainIsSelfContainedPipeline) {
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
    // PiD decode (Z-Image reuses the FLUX decoder) needs both a PiD decoder and the Gemma-2 caption encoder.
    if (params.pidMode !== 'off') {
      if (!params.pidDecoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
      }
      if (!params.gemma2EncoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
      }
    }
    // PiD decode (Z-Image reuses the FLUX decoder) needs both a PiD decoder and the Gemma-2 caption encoder.
    if (params.pidMode !== 'off') {
      if (!params.pidDecoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
      }
      if (!params.gemma2EncoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
      }
    }
  }

  if (model?.base === 'krea-2' && model.format !== 'diffusers') {
    // Non-diffusers Krea-2 (single-file checkpoint / GGUF) ships only the transformer, so a standalone
    // VAE and Qwen3-VL encoder must be selected. Diffusers models bundle them, so they're optional there.
    if (!params.krea2VaeModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noKrea2VaeModelSelected') });
    }
    if (!params.krea2Qwen3VlEncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noKrea2Qwen3VlEncoderModelSelected') });
    }
  }

  if (
    model?.base === 'krea-2' &&
    params.krea2RebalanceEnabled &&
    !isValidKrea2RebalanceWeights(params.krea2RebalanceWeights)
  ) {
    // The rebalance weights are free text forwarded straight to the backend; block generation before an
    // invalid string (wrong count / nonnumeric / nan / inf) reaches the failing _parse_weights().
    reasons.push({ content: i18n.t('parameters.invoke.krea2RebalanceWeightsInvalid') });
  }

  if (model?.base === 'anima') {
    if (!params.animaVaeModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noAnimaVaeModelSelected') });
    }
    if (!params.animaQwen3EncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noAnimaQwen3EncoderModelSelected') });
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

  if (model && !isExternalApiModelConfig(model) && SUPPORTS_REF_IMAGES_BASE_MODELS.includes(model.base)) {
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

export const getReasonsWhyCannotEnqueueCanvasTab = (arg: {
  isConnected: boolean;
  model: MainOrExternalModelConfig | null | undefined;
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
  hasFlux2DiffusersVaeSource: boolean;
  hasFlux2DiffusersQwen3Source: boolean;
  hasFlux2DevDiffusersSource: boolean;
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
    hasFlux2DiffusersVaeSource,
    hasFlux2DiffusersQwen3Source,
    hasFlux2DevDiffusersSource,
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

  if (!model) {
    // nothing else to validate
  } else if (isExternalApiModelConfig(model)) {
    // external models don't require local sub-models
  } else if (model.base === 'flux') {
    // A complete SDNQ FLUX.1 pipeline install ships its own T5, CLIP and VAE, and the model loader
    // node falls back to them, so requiring the standalone selections here would keep that path
    // unreachable from the UI. Anything else (single-file, GGUF, BnB) still needs all three.
    const mainSuppliesComponents = isSelfContainedSDNQFlux1Pipeline(model);
    if (!mainSuppliesComponents) {
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

    const { bbox } = canvas;
    // In PiD native mode the bbox is the 4x target, so it must snap to a larger grid (16 * 4) for bbox / 4 to land
    // on the FLUX grid. getPidScale returns 1 for off/fit, leaving the normal 16px grid.
    const gridSize = getGridSize('flux', getPidScale(params.pidMode));

    if (params.pidMode !== 'off') {
      if (!params.pidDecoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
      }
      if (!params.gemma2EncoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
      }
      // PiD decodes at 4x the generation resolution; "Scale Before Processing" would inflate the generation
      // size and blow up the decode. Require it to be off (None) so generation == bbox.
      if (bbox.scaleMethod !== 'none') {
        reasons.push({ content: i18n.t('parameters.invoke.pidScaleBeforeProcessingMustBeOff') });
      }
    }

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
    // A FLUX.2 model is a self-sufficient source when its config exposes the diffusers-style
    // submodels. Plain Diffusers pipelines always do; an SDNQ pipeline qualifies only when it ships
    // all of them — a truthy submodels dict is not enough, since a partial pipeline may expose only
    // the transformer and the backend would then request missing fixed subfolders. Mirrors the
    // generate-tab check so both tabs behave identically.
    const mainIsPipeline =
      model.format === 'diffusers' ||
      ((model as { format?: unknown }).format === 'sdnq_quantized' && isSelfContainedSDNQPipeline(model));
    // VAE is shared across variants, but the text encoder requires a variant-matching diffusers model.
    if (!mainIsPipeline) {
      if ('variant' in model && model.variant === 'dev') {
        if (!params.flux2VaeModel && !hasFlux2DevDiffusersSource) {
          reasons.push({ content: i18n.t('parameters.invoke.noFlux2DevVaeModelSelected') });
        }
        if (!params.flux2DevMistralEncoderModel && !hasFlux2DevDiffusersSource) {
          reasons.push({ content: i18n.t('parameters.invoke.noFlux2DevMistralEncoderModelSelected') });
        }
      } else {
        if (!params.flux2VaeModel && !hasFlux2DiffusersVaeSource) {
          reasons.push({ content: i18n.t('parameters.invoke.noFlux2KleinVaeModelSelected') });
        }
        if (!params.kleinQwen3EncoderModel && !hasFlux2DiffusersQwen3Source) {
          reasons.push({ content: i18n.t('parameters.invoke.noFlux2KleinQwen3EncoderModelSelected') });
        }
      }
    }

    const { bbox } = canvas;
    // FLUX.2 uses the same 16px grid as FLUX.1. In PiD native mode the bbox is the 4x target, so it must snap to
    // a larger grid (16 * 4) for bbox / 4 to land on the FLUX grid. getPidScale returns 1 for off/fit.
    const gridSize = getGridSize('flux2', getPidScale(params.pidMode));

    if (params.pidMode !== 'off') {
      if (!params.pidDecoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
      }
      if (!params.gemma2EncoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
      }
      // PiD decodes at 4x the generation resolution; "Scale Before Processing" would inflate the generation
      // size and blow up the decode. Require it to be off (None) so generation == bbox.
      if (bbox.scaleMethod !== 'none') {
        reasons.push({ content: i18n.t('parameters.invoke.pidScaleBeforeProcessingMustBeOff') });
      }
    }

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

  if (model?.base === 'sd-3' && params.pidMode !== 'off') {
    // PiD decode on the Canvas: needs the decoder + Gemma-2 encoder, and "Scale Before Processing" must be off
    // (PiD decodes at 4x the generation resolution; scaling would inflate the generation size and blow up the decode).
    if (!params.pidDecoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
    }
    if (!params.gemma2EncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
    }
    if (canvas.bbox.scaleMethod !== 'none') {
      reasons.push({ content: i18n.t('parameters.invoke.pidScaleBeforeProcessingMustBeOff') });
    }
    // Native mode generates at bbox/4, so the bbox must be a multiple of the PiD-scaled grid (grid*4) for
    // bbox/4 to land on the SD3 grid; without this a 1040px bbox silently becomes a 256px generation.
    const gridSize = getGridSize('sd-3', getPidScale(params.pidMode));
    if (canvas.bbox.rect.width % gridSize !== 0) {
      reasons.push({
        content: i18n.t('parameters.invoke.modelIncompatibleBboxWidth', {
          model: 'SD3',
          width: canvas.bbox.rect.width,
          multiple: gridSize,
        }),
      });
    }
    if (canvas.bbox.rect.height % gridSize !== 0) {
      reasons.push({
        content: i18n.t('parameters.invoke.modelIncompatibleBboxHeight', {
          model: 'SD3',
          height: canvas.bbox.rect.height,
          multiple: gridSize,
        }),
      });
    }
  }

  if (model?.base === 'sdxl' && params.pidMode !== 'off') {
    // PiD decode on the Canvas: decoder + Gemma-2 encoder required, "Scale Before Processing" off, and not
    // compatible with the SDXL Refiner.
    if (!params.pidDecoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
    }
    if (!params.gemma2EncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
    }
    if (params.refinerModel) {
      reasons.push({ content: i18n.t('parameters.invoke.pidIncompatibleWithRefiner') });
    }
    if (canvas.bbox.scaleMethod !== 'none') {
      reasons.push({ content: i18n.t('parameters.invoke.pidScaleBeforeProcessingMustBeOff') });
    }
    // Native mode generates at bbox/4, so the bbox must be a multiple of the PiD-scaled grid (grid*4) for
    // bbox/4 to land on the SDXL grid; without this a 1040px bbox silently becomes a 256px generation.
    const gridSize = getGridSize('sdxl', getPidScale(params.pidMode));
    if (canvas.bbox.rect.width % gridSize !== 0) {
      reasons.push({
        content: i18n.t('parameters.invoke.modelIncompatibleBboxWidth', {
          model: 'SDXL',
          width: canvas.bbox.rect.width,
          multiple: gridSize,
        }),
      });
    }
    if (canvas.bbox.rect.height % gridSize !== 0) {
      reasons.push({
        content: i18n.t('parameters.invoke.modelIncompatibleBboxHeight', {
          model: 'SDXL',
          height: canvas.bbox.rect.height,
          multiple: gridSize,
        }),
      });
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

  if (model?.base === 'qwen-image') {
    const { bbox } = canvas;
    // In PiD native mode the bbox is the 4x target, so it must snap to a larger grid (16 * 4) for bbox / 4 to land
    // on the Qwen grid. getPidScale returns 1 for off/fit, leaving the normal 16px grid.
    const gridSize = getGridSize('qwen-image', getPidScale(params.pidMode));

    if (params.pidMode !== 'off') {
      if (!params.pidDecoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
      }
      if (!params.gemma2EncoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
      }
      if (bbox.scaleMethod !== 'none') {
        reasons.push({ content: i18n.t('parameters.invoke.pidScaleBeforeProcessingMustBeOff') });
      }
    }

    if (bbox.scaleMethod === 'none') {
      if (bbox.rect.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxWidth', {
            model: 'Qwen Image Edit',
            width: bbox.rect.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.rect.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxHeight', {
            model: 'Qwen Image Edit',
            height: bbox.rect.height,
            multiple: gridSize,
          }),
        });
      }
    } else {
      if (bbox.scaledSize.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxWidth', {
            model: 'Qwen Image Edit',
            width: bbox.scaledSize.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.scaledSize.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxHeight', {
            model: 'Qwen Image Edit',
            height: bbox.scaledSize.height,
            multiple: gridSize,
          }),
        });
      }
    }
  }

  if (model?.base === 'ernie-image') {
    // ERNIE-Image requires bbox dimensions that are multiples of 16 (enforced by ernie_image_denoise).
    const { bbox } = canvas;
    const gridSize = getGridSize('ernie-image');

    if (bbox.scaleMethod === 'none') {
      if (bbox.rect.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxWidth', {
            model: 'ERNIE-Image',
            width: bbox.rect.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.rect.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxHeight', {
            model: 'ERNIE-Image',
            height: bbox.rect.height,
            multiple: gridSize,
          }),
        });
      }
    } else {
      if (bbox.scaledSize.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxWidth', {
            model: 'ERNIE-Image',
            width: bbox.scaledSize.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.scaledSize.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxHeight', {
            model: 'ERNIE-Image',
            height: bbox.scaledSize.height,
            multiple: gridSize,
          }),
        });
      }
    }
  }

  if (model?.base === 'ideogram-4') {
    // Ideogram 4 requires bbox dimensions that are multiples of 16 (enforced by ideogram4_denoise).
    const { bbox } = canvas;
    const gridSize = getGridSize('ideogram-4');

    if (bbox.scaleMethod === 'none') {
      if (bbox.rect.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxWidth', {
            model: 'Ideogram 4',
            width: bbox.rect.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.rect.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxHeight', {
            model: 'Ideogram 4',
            height: bbox.rect.height,
            multiple: gridSize,
          }),
        });
      }
    } else {
      if (bbox.scaledSize.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxWidth', {
            model: 'Ideogram 4',
            width: bbox.scaledSize.width,
            multiple: gridSize,
          }),
        });
      }
      if (bbox.scaledSize.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleScaledBboxHeight', {
            model: 'Ideogram 4',
            height: bbox.scaledSize.height,
            multiple: gridSize,
          }),
        });
      }
    }
  }

  if (model?.base === 'qwen-image' && model.format === 'gguf_quantized') {
    // GGUF needs sources for VAE + encoder. Each can come from either a standalone
    // model or the Component Source (Diffusers).
    const hasVaeSource = params.qwenImageVaeModel !== null || params.qwenImageComponentSource !== null;
    const hasEncoderSource = params.qwenImageQwenVLEncoderModel !== null || params.qwenImageComponentSource !== null;
    if (!hasVaeSource || !hasEncoderSource) {
      reasons.push({ content: i18n.t('parameters.invoke.noQwenImageComponentSourceSelected') });
    }
  }

  if (model && isWanSingleFileMainModelConfig(model)) {
    pushWanSingleFileReasons(params, reasons);
  }

  if (model?.base === 'z-image') {
    // An SDNQ-quantized Z-Image pipeline install is self-contained: it ships the VAE and Qwen3
    // encoder (text_encoder + tokenizer) as submodels of the main model, so no separate component
    // source is required. A truthy submodels dict is not enough — a partial pipeline may expose only
    // some submodels — so require every one the loader needs. Single-file / GGUF Z-Image models
    // don't have submodels and still need a standalone VAE + Qwen3 (or a Qwen3 Source model).
    const mainIsSelfContainedPipeline =
      (model as { format?: unknown }).format === 'sdnq_quantized' && isSelfContainedSDNQPipeline(model);
    if (!mainIsSelfContainedPipeline) {
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
    // PiD decode on the Canvas: decoder + Gemma-2 encoder required, and "Scale Before Processing" must be off.
    if (params.pidMode !== 'off') {
      if (!params.pidDecoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
      }
      if (!params.gemma2EncoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
      }
      if (canvas.bbox.scaleMethod !== 'none') {
        reasons.push({ content: i18n.t('parameters.invoke.pidScaleBeforeProcessingMustBeOff') });
      }
      // Native mode generates at bbox/4, so the bbox must be a multiple of the PiD-scaled grid (grid*4) for
      // bbox/4 to land on the grid; without this a 1040px bbox silently becomes a 256px generation.
      const gridSize = getGridSize('z-image', getPidScale(params.pidMode));
      if (canvas.bbox.rect.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxWidth', {
            model: 'Z-Image',
            width: canvas.bbox.rect.width,
            multiple: gridSize,
          }),
        });
      }
      if (canvas.bbox.rect.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxHeight', {
            model: 'Z-Image',
            height: canvas.bbox.rect.height,
            multiple: gridSize,
          }),
        });
      }
    }
    // PiD decode on the Canvas: decoder + Gemma-2 encoder required, and "Scale Before Processing" must be off.
    if (params.pidMode !== 'off') {
      if (!params.pidDecoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
      }
      if (!params.gemma2EncoderModel) {
        reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
      }
      if (canvas.bbox.scaleMethod !== 'none') {
        reasons.push({ content: i18n.t('parameters.invoke.pidScaleBeforeProcessingMustBeOff') });
      }
      // Native mode generates at bbox/4, so the bbox must be a multiple of the PiD-scaled grid (grid*4) for
      // bbox/4 to land on the grid; without this a 1040px bbox silently becomes a 256px generation.
      const gridSize = getGridSize('z-image', getPidScale(params.pidMode));
      if (canvas.bbox.rect.width % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxWidth', {
            model: 'Z-Image',
            width: canvas.bbox.rect.width,
            multiple: gridSize,
          }),
        });
      }
      if (canvas.bbox.rect.height % gridSize !== 0) {
        reasons.push({
          content: i18n.t('parameters.invoke.modelIncompatibleBboxHeight', {
            model: 'Z-Image',
            height: canvas.bbox.rect.height,
            multiple: gridSize,
          }),
        });
      }
    }
  }

  if (model?.base === 'krea-2' && model.format !== 'diffusers') {
    // Non-diffusers Krea-2 (single-file checkpoint / GGUF) ships only the transformer, so a standalone
    // VAE and Qwen3-VL encoder must be selected. Diffusers models bundle them, so they're optional there.
    if (!params.krea2VaeModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noKrea2VaeModelSelected') });
    }
    if (!params.krea2Qwen3VlEncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noKrea2Qwen3VlEncoderModelSelected') });
    }
  }

  if (
    model?.base === 'krea-2' &&
    params.krea2RebalanceEnabled &&
    !isValidKrea2RebalanceWeights(params.krea2RebalanceWeights)
  ) {
    // The rebalance weights are free text forwarded straight to the backend; block generation before an
    // invalid string (wrong count / nonnumeric / nan / inf) reaches the failing _parse_weights().
    reasons.push({ content: i18n.t('parameters.invoke.krea2RebalanceWeightsInvalid') });
  }

  if (model?.base === 'anima') {
    if (!params.animaVaeModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noAnimaVaeModelSelected') });
    }
    if (!params.animaQwen3EncoderModel) {
      reasons.push({ content: i18n.t('parameters.invoke.noAnimaQwen3EncoderModelSelected') });
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
    const problems = getControlLayerWarnings(controlLayer, model, enabledControlLayers);

    if (problems.length) {
      const content = upperFirst(problems.map((p) => i18n.t(p)).join(', '));
      reasons.push({ prefix, content });
    }
  });

  if (model && !isExternalApiModelConfig(model) && SUPPORTS_REF_IMAGES_BASE_MODELS.includes(model.base)) {
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
