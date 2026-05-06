import type { RootState } from 'app/store/store';
import { roundDownToMultiple } from 'common/util/roundDownToMultiple';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { isHrfSupportedBase, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import type { GenerationMode, LoRA } from 'features/controlLayers/store/types';
import type { BaseModelType } from 'features/nodes/types/common';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { addLoRAs } from 'features/nodes/util/graph/generation/addLoRAs';
import { addSDXLLoRAs } from 'features/nodes/util/graph/generation/addSDXLLoRAs';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { DenoiseLatentsNodes, LatentToImageNodes } from 'features/nodes/util/graph/types';
import { getGridSize } from 'features/parameters/util/optimalDimension';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { AnyInvocation, AnyInvocationInputField, AnyInvocationOutputField, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

type AddHighResFixArg = {
  g: Graph;
  state: RootState;
  generationMode: GenerationMode;
  denoise: Invocation<DenoiseLatentsNodes>;
  l2i: Invocation<LatentToImageNodes>;
  noise?: Invocation<'noise'>;
  seed: Invocation<'integer'>;
};

type HrfModelLoader = Invocation<'main_model_loader'> | Invocation<'sdxl_model_loader'>;
type HrfVaeSource = HrfModelLoader | Invocation<'seamless'>;
type FreshHrfModelInputs = {
  modelLoader: HrfModelLoader;
  vaeSource: HrfVaeSource;
};
type ConditioningSource = {
  node_id: string;
  field: AnyInvocationOutputField;
};

const SKIPPED_DENOISE_INPUT_FIELDS = new Set(['latents', 'noise']);
const FRESH_HRF_MODEL_INPUT_FIELDS = new Set(['unet', 'positive_conditioning', 'negative_conditioning']);
const TILED_DENOISE_INPUT_FIELDS = new Set(['positive_conditioning', 'negative_conditioning', 'unet']);
const SKIPPED_LATENT_TO_IMAGE_INPUT_FIELDS = new Set(['latents', 'metadata']);
const LATENT_HRF_L2I_TILE_SIZE = 1024;

const getHighResFixFinalDimensions = (state: RootState) => {
  const params = selectParamsSlice(state);
  const gridSize = getGridSize(params.model?.base as BaseModelType | undefined);

  return {
    width: Math.max(roundDownToMultiple(params.dimensions.width * params.hrfScale, gridSize), 64),
    height: Math.max(roundDownToMultiple(params.dimensions.height * params.hrfScale, gridSize), 64),
  };
};

const getHighResFixDenoisingStartAndEnd = (state: RootState): { denoising_start: number; denoising_end: number } => {
  const params = selectParamsSlice(state);
  const model = params.model;
  const { hrfStrength, optimizedDenoisingEnabled } = params;

  switch (model?.base) {
    case 'sd-3':
    case 'flux':
    case 'flux2': {
      if (model.base === 'flux' && 'variant' in model && model.variant === 'dev_fill') {
        return { denoising_start: 0, denoising_end: 1 };
      }

      const exponent = optimizedDenoisingEnabled ? 0.2 : 1;
      return { denoising_start: 1 - hrfStrength ** exponent, denoising_end: 1 };
    }
    case 'anima':
    case 'sd-1':
    case 'sd-2':
    case 'sdxl':
    case 'cogview4':
    case 'qwen-image':
    case 'z-image': {
      return { denoising_start: 1 - hrfStrength, denoising_end: 1 };
    }
    default: {
      assert(false, `Unsupported base for high resolution fix: ${model?.base}`);
    }
  }
};

const shouldApplyHighResFix = (state: RootState, generationMode: GenerationMode) => {
  const params = selectParamsSlice(state);
  const model = params.model;

  return (
    selectActiveTab(state) === 'generate' &&
    generationMode === 'txt2img' &&
    params.hrfEnabled &&
    model !== null &&
    isHrfSupportedBase(model.base) &&
    !params.refinerModel
  );
};

const shouldWriteDisabledMetadata = (state: RootState, generationMode: GenerationMode) => {
  return selectActiveTab(state) === 'generate' && generationMode === 'txt2img';
};

const cloneSDXLCompelPromptForFinalDimensions = (
  g: Graph,
  node: Invocation<'sdxl_compel_prompt'>,
  finalDimensions: { width: number; height: number }
) => {
  const clone = g.addNode({
    ...node,
    id: getPrefixedId(`hrf_${node.id.split(':')[0]}`),
    original_width: finalDimensions.width,
    original_height: finalDimensions.height,
    target_width: finalDimensions.width,
    target_height: finalDimensions.height,
  });

  for (const edge of g.getEdgesTo(node)) {
    g.addEdgeFromObj({
      source: { ...edge.source },
      destination: { node_id: clone.id, field: edge.destination.field },
    });
  }

  return clone;
};

const cloneSDXLConditioningForFinalDimensions = (
  g: Graph,
  sourceNodeId: string,
  finalDimensions: { width: number; height: number },
  outputType: 'collection' | 'single'
) => {
  const sourceNode = g.getNode(sourceNodeId);

  if (sourceNode.type === 'sdxl_compel_prompt') {
    return {
      nodeId: cloneSDXLCompelPromptForFinalDimensions(g, sourceNode, finalDimensions).id,
      field: 'conditioning',
    };
  }

  if (sourceNode.type !== 'collect') {
    return null;
  }

  const itemEdges = g.getEdgesTo(sourceNode).filter((edge) => edge.destination.field === 'item');
  const hasSDXLConditioning = itemEdges.some((edge) => g.getNode(edge.source.node_id).type === 'sdxl_compel_prompt');

  if (!hasSDXLConditioning) {
    if (outputType === 'single' && itemEdges.length === 1) {
      return { nodeId: itemEdges[0]!.source.node_id, field: itemEdges[0]!.source.field };
    }

    return null;
  }

  if (outputType === 'single') {
    if (itemEdges.length !== 1) {
      return null;
    }

    const edge = itemEdges[0]!;
    const itemNode = g.getNode(edge.source.node_id);
    const source =
      itemNode.type === 'sdxl_compel_prompt'
        ? cloneSDXLCompelPromptForFinalDimensions(g, itemNode, finalDimensions)
        : itemNode;

    return { nodeId: source.id, field: edge.source.field };
  }

  const collect = g.addNode({
    type: 'collect',
    id: getPrefixedId(`hrf_${sourceNode.id.split(':')[0]}_conditioning_collect`),
  });

  for (const edge of itemEdges) {
    const itemNode = g.getNode(edge.source.node_id);
    const source =
      itemNode.type === 'sdxl_compel_prompt'
        ? cloneSDXLCompelPromptForFinalDimensions(g, itemNode, finalDimensions)
        : itemNode;

    g.addEdgeFromObj({
      source: { node_id: source.id, field: edge.source.field },
      destination: { node_id: collect.id, field: 'item' },
    });
  }

  return { nodeId: collect.id, field: 'collection' };
};

const copyDenoiseInputs = (
  g: Graph,
  from: AnyInvocation,
  to: AnyInvocation,
  finalDimensions: { width: number; height: number },
  allowedInputFields?: Set<string>,
  skippedInputFields?: Set<string>
) => {
  for (const edge of g.getEdgesTo(from)) {
    if (SKIPPED_DENOISE_INPUT_FIELDS.has(edge.destination.field)) {
      continue;
    }
    if (skippedInputFields?.has(edge.destination.field)) {
      continue;
    }
    if (allowedInputFields && !allowedInputFields.has(edge.destination.field)) {
      continue;
    }

    const finalSizeConditioning = ['positive_conditioning', 'negative_conditioning'].includes(edge.destination.field)
      ? cloneSDXLConditioningForFinalDimensions(
          g,
          edge.source.node_id,
          finalDimensions,
          to.type === 'tiled_multi_diffusion_denoise_latents' ? 'single' : 'collection'
        )
      : null;

    g.addEdgeFromObj({
      source: finalSizeConditioning
        ? {
            node_id: finalSizeConditioning.nodeId,
            field: finalSizeConditioning.field as AnyInvocationOutputField,
          }
        : { ...edge.source },
      destination: { node_id: to.id, field: edge.destination.field as AnyInvocationInputField },
    });
  }
};

const copyInputEdges = (g: Graph, from: AnyInvocation, to: AnyInvocation, skippedInputFields: Set<string>) => {
  for (const edge of g.getEdgesTo(from)) {
    if (skippedInputFields.has(edge.destination.field)) {
      continue;
    }
    g.addEdgeFromObj({
      source: { ...edge.source },
      destination: { node_id: to.id, field: edge.destination.field },
    });
  }
};

const hasUnsupportedTiledDenoiseInputs = (g: Graph, denoise: Invocation<'denoise_latents'>) => {
  return g.getEdgesTo(denoise).some((edge) => {
    const field = edge.destination.field;
    if (SKIPPED_DENOISE_INPUT_FIELDS.has(field)) {
      return false;
    }

    if (!TILED_DENOISE_INPUT_FIELDS.has(field)) {
      return true;
    }

    if (['positive_conditioning', 'negative_conditioning'].includes(field)) {
      const sourceNode = g.getNode(edge.source.node_id);
      if (sourceNode.type === 'collect') {
        const itemEdges = g.getEdgesTo(sourceNode).filter((itemEdge) => itemEdge.destination.field === 'item');
        return itemEdges.length !== 1;
      }
    }

    return false;
  });
};

const addTileControlNet = (
  g: Graph,
  hrfDenoise: AnyInvocation,
  imageSource: Invocation<'unsharp_mask'>,
  tileControlNetModel: NonNullable<ReturnType<typeof selectParamsSlice>['hrfTileControlNetModel']>,
  tileControlWeight: number,
  tileControlEnd: number
) => {
  const controlNet = g.addNode({
    id: getPrefixedId('hrf_controlnet'),
    type: 'controlnet',
    control_model: tileControlNetModel,
    control_mode: 'balanced',
    resize_mode: 'just_resize',
    control_weight: tileControlWeight,
    begin_step_percent: 0,
    end_step_percent: tileControlEnd,
  });

  g.addEdge(imageSource, 'image', controlNet, 'image');
  g.addEdgeFromObj({
    source: { node_id: controlNet.id, field: 'control' },
    destination: { node_id: hrfDenoise.id, field: 'control' },
  });
};

const getConditioningSources = (
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  field: 'positive_conditioning' | 'negative_conditioning'
): ConditioningSource[] => {
  const edge = g.getEdgesTo(denoise).find((edge) => edge.destination.field === field);
  assert(edge, `Missing ${field} edge for HRF second pass`);

  const sourceNode = g.getNode(edge.source.node_id);
  if (sourceNode.type !== 'collect') {
    return [{ node_id: edge.source.node_id, field: edge.source.field as AnyInvocationOutputField }];
  }

  const itemEdges = g.getEdgesTo(sourceNode).filter((edge) => edge.destination.field === 'item');
  assert(itemEdges.length > 0, `HRF second pass expects at least one ${field} conditioning source`);

  return itemEdges.map((edge) => ({
    node_id: edge.source.node_id,
    field: edge.source.field as AnyInvocationOutputField,
  }));
};

const connectConditioningToDenoise = (
  g: Graph,
  hrfDenoise: Invocation<'denoise_latents'> | Invocation<'tiled_multi_diffusion_denoise_latents'>,
  field: 'positive_conditioning' | 'negative_conditioning',
  conditioning: Array<Invocation<'compel'> | Invocation<'sdxl_compel_prompt'>>
) => {
  assert(conditioning.length > 0, `HRF second pass expects at least one ${field} conditioning node`);

  if (conditioning.length === 1) {
    g.addEdgeFromObj({
      source: { node_id: conditioning[0]!.id, field: 'conditioning' },
      destination: { node_id: hrfDenoise.id, field },
    });
    return;
  }

  assert(hrfDenoise.type === 'denoise_latents', 'Tiled HRF second pass supports only single conditioning inputs');

  const collect = g.addNode({
    type: 'collect',
    id: getPrefixedId(`hrf_${field}_collect`),
  });

  for (const cond of conditioning) {
    g.addEdge(cond, 'conditioning', collect, 'item');
  }

  g.addEdgeFromObj({
    source: { node_id: collect.id, field: 'collection' },
    destination: { node_id: hrfDenoise.id, field },
  });
};

const cloneSD1ConditioningSources = (
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  field: 'positive_conditioning' | 'negative_conditioning'
) => {
  return getConditioningSources(g, denoise, field).map((source) => {
    const sourceNode = g.getNode(source.node_id);
    assert(sourceNode.type === 'compel', 'SD1 HRF refinement requires SD1 conditioning');

    const clone = g.addNode({
      ...sourceNode,
      id: getPrefixedId(`hrf_${sourceNode.id.split(':')[0]}`),
    });
    copyInputEdges(g, sourceNode, clone, new Set(['clip']));
    return clone;
  });
};

const cloneSDXLConditioningSources = (
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  field: 'positive_conditioning' | 'negative_conditioning',
  finalDimensions: { width: number; height: number }
) => {
  return getConditioningSources(g, denoise, field).map((source) => {
    const sourceNode = g.getNode(source.node_id);
    assert(sourceNode.type === 'sdxl_compel_prompt', 'SDXL HRF refinement requires SDXL conditioning');

    const clone = g.addNode({
      ...sourceNode,
      id: getPrefixedId(`hrf_${sourceNode.id.split(':')[0]}`),
      original_width: finalDimensions.width,
      original_height: finalDimensions.height,
      target_width: finalDimensions.width,
      target_height: finalDimensions.height,
    });
    copyInputEdges(g, sourceNode, clone, new Set(['clip', 'clip2']));
    return clone;
  });
};

const findOriginalSeamlessNode = (g: Graph, denoise: Invocation<'denoise_latents'>) => {
  let current: AnyInvocation = denoise;
  const visited = new Set<string>();

  while (!visited.has(current.id)) {
    visited.add(current.id);
    const unetEdge = g.getEdgesTo(current).find((edge) => edge.destination.field === 'unet');
    if (!unetEdge) {
      return null;
    }

    const sourceNode = g.getNode(unetEdge.source.node_id);
    if (sourceNode.type === 'seamless') {
      return sourceNode;
    }

    current = sourceNode;
  }

  return null;
};

const addFreshHrfSeamless = (
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  hrfDenoise: Invocation<'denoise_latents'> | Invocation<'tiled_multi_diffusion_denoise_latents'>,
  modelLoader: HrfModelLoader,
  useHrfModelVae: boolean
) => {
  const originalSeamless = findOriginalSeamlessNode(g, denoise);
  if (!originalSeamless) {
    return null;
  }

  const seamless = g.addNode({
    ...originalSeamless,
    id: getPrefixedId('hrf_seamless'),
  });

  g.addEdge(modelLoader, 'unet', seamless, 'unet');

  if (useHrfModelVae) {
    g.addEdge(modelLoader, 'vae', seamless, 'vae');
  } else {
    const originalVaeEdge = g.getEdgesTo(originalSeamless).find((edge) => edge.destination.field === 'vae');
    if (originalVaeEdge) {
      g.addEdgeFromObj({
        source: { ...originalVaeEdge.source },
        destination: { node_id: seamless.id, field: 'vae' },
      });
    } else {
      g.addEdge(modelLoader, 'vae', seamless, 'vae');
    }
  }

  g.addEdge(seamless, 'unet', hrfDenoise, 'unet');
  return seamless;
};

const getEnabledHrfLoRAs = (state: RootState): LoRA[] | null => {
  const params = selectParamsSlice(state);

  if (params.hrfLoraMode === 'none') {
    return null;
  }

  if (params.hrfLoraMode === 'dedicated') {
    return getEnabledDedicatedHrfLoRAs(state);
  }

  return state.loras.loras.filter((lora) => lora.isEnabled);
};

const getEnabledDedicatedHrfLoRAs = (state: RootState): LoRA[] => {
  const params = selectParamsSlice(state);
  const effectiveHrfBase = params.hrfModel?.base ?? params.model?.base;
  if (!effectiveHrfBase) {
    return [];
  }
  return params.hrfLoras.filter((lora) => lora.isEnabled && lora.model.base === effectiveHrfBase);
};

const getHrfLoRAMetadata = (state: RootState) => {
  const params = selectParamsSlice(state);
  if (params.hrfLoraMode !== 'dedicated') {
    return undefined;
  }

  return getEnabledDedicatedHrfLoRAs(state).map((lora) => ({
    model: zModelIdentifierField.parse(lora.model),
    weight: lora.weight,
  }));
};

const addFreshSD1HrfModelInputs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  hrfDenoise: Invocation<'denoise_latents'> | Invocation<'tiled_multi_diffusion_denoise_latents'>
): FreshHrfModelInputs => {
  const params = selectParamsSlice(state);
  const hrfModel = params.hrfModel ?? params.model;
  assert(hrfModel?.base === 'sd-1', 'SD1 HRF refinement model must be an SD1 model');

  const modelLoader = g.addNode({
    type: 'main_model_loader',
    id: getPrefixedId('hrf_sd1_model_loader'),
    model: hrfModel,
  });
  const clipSkip = g.addNode({
    type: 'clip_skip',
    id: getPrefixedId('hrf_clip_skip'),
    skipped_layers: params.clipSkip,
  });

  const positiveConditioning = cloneSD1ConditioningSources(g, denoise, 'positive_conditioning');
  const negativeConditioning = cloneSD1ConditioningSources(g, denoise, 'negative_conditioning');
  const hrfSeamless = addFreshHrfSeamless(g, denoise, hrfDenoise, modelLoader, params.hrfModel !== null);

  g.addEdge(modelLoader, 'clip', clipSkip, 'clip');
  for (const cond of positiveConditioning) {
    g.addEdge(clipSkip, 'clip', cond, 'clip');
  }
  for (const cond of negativeConditioning) {
    g.addEdge(clipSkip, 'clip', cond, 'clip');
  }
  if (!hrfSeamless) {
    g.addEdgeFromObj({
      source: { node_id: modelLoader.id, field: 'unet' },
      destination: { node_id: hrfDenoise.id, field: 'unet' },
    });
  }
  connectConditioningToDenoise(g, hrfDenoise, 'positive_conditioning', positiveConditioning);
  connectConditioningToDenoise(g, hrfDenoise, 'negative_conditioning', negativeConditioning);

  const enabledLoRAs = getEnabledHrfLoRAs(state);
  if (enabledLoRAs?.length) {
    addLoRAs(
      state,
      g,
      hrfDenoise,
      modelLoader,
      hrfSeamless,
      clipSkip,
      positiveConditioning[0]!,
      negativeConditioning[0]!,
      {
        loras: enabledLoRAs,
        idPrefix: 'hrf',
        metadataKey: params.hrfLoraMode === 'dedicated' ? 'hrf_loras' : 'loras',
        extraPositiveConditioning: positiveConditioning.slice(1),
        extraNegativeConditioning: negativeConditioning.slice(1),
      }
    );
  }

  return { modelLoader, vaeSource: hrfSeamless ?? modelLoader };
};

const addFreshSDXLHrfModelInputs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  hrfDenoise: Invocation<'denoise_latents'> | Invocation<'tiled_multi_diffusion_denoise_latents'>,
  finalDimensions: { width: number; height: number }
): FreshHrfModelInputs => {
  const params = selectParamsSlice(state);
  const hrfModel = params.hrfModel ?? params.model;
  assert(hrfModel?.base === 'sdxl', 'SDXL HRF refinement model must be an SDXL model');

  const modelLoader = g.addNode({
    type: 'sdxl_model_loader',
    id: getPrefixedId('hrf_sdxl_model_loader'),
    model: hrfModel,
  });

  const positiveConditioning = cloneSDXLConditioningSources(g, denoise, 'positive_conditioning', finalDimensions);
  const negativeConditioning = cloneSDXLConditioningSources(g, denoise, 'negative_conditioning', finalDimensions);
  const hrfSeamless = addFreshHrfSeamless(g, denoise, hrfDenoise, modelLoader, params.hrfModel !== null);

  for (const cond of positiveConditioning) {
    g.addEdge(modelLoader, 'clip', cond, 'clip');
    g.addEdge(modelLoader, 'clip2', cond, 'clip2');
  }
  for (const cond of negativeConditioning) {
    g.addEdge(modelLoader, 'clip', cond, 'clip');
    g.addEdge(modelLoader, 'clip2', cond, 'clip2');
  }
  if (!hrfSeamless) {
    g.addEdgeFromObj({
      source: { node_id: modelLoader.id, field: 'unet' },
      destination: { node_id: hrfDenoise.id, field: 'unet' },
    });
  }
  connectConditioningToDenoise(g, hrfDenoise, 'positive_conditioning', positiveConditioning);
  connectConditioningToDenoise(g, hrfDenoise, 'negative_conditioning', negativeConditioning);

  const enabledLoRAs = getEnabledHrfLoRAs(state);
  if (enabledLoRAs?.length) {
    addSDXLLoRAs(state, g, hrfDenoise, modelLoader, hrfSeamless, positiveConditioning[0]!, negativeConditioning[0]!, {
      loras: enabledLoRAs,
      idPrefix: 'hrf',
      metadataKey: params.hrfLoraMode === 'dedicated' ? 'hrf_loras' : 'loras',
      extraPositiveConditioning: positiveConditioning.slice(1),
      extraNegativeConditioning: negativeConditioning.slice(1),
    });
  }

  return { modelLoader, vaeSource: hrfSeamless ?? modelLoader };
};

const addFreshHrfModelInputs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  hrfDenoise: Invocation<'denoise_latents'> | Invocation<'tiled_multi_diffusion_denoise_latents'>,
  finalDimensions: { width: number; height: number }
): FreshHrfModelInputs => {
  const params = selectParamsSlice(state);

  if (params.model?.base === 'sdxl') {
    return addFreshSDXLHrfModelInputs(state, g, denoise, hrfDenoise, finalDimensions);
  }

  return addFreshSD1HrfModelInputs(state, g, denoise, hrfDenoise);
};

const addLatentHighResFix = ({
  g,
  state,
  denoise,
  l2i,
  noise,
  seed,
}: AddHighResFixArg): Invocation<LatentToImageNodes> => {
  const params = selectParamsSlice(state);
  const finalDimensions = getHighResFixFinalDimensions(state);
  const { denoising_start, denoising_end } = getHighResFixDenoisingStartAndEnd(state);

  const resizeLatents = g.addNode({
    id: getPrefixedId('hrf_resize_latents'),
    type: 'lresize',
    ...finalDimensions,
    mode: params.hrfLatentInterpolationMode,
    antialias: false,
  });

  const hrfDenoise = g.addNode({
    ...denoise,
    id: getPrefixedId(`hrf_${denoise.type}`),
    denoising_start,
    denoising_end,
    ...(denoise.type === 'denoise_latents' ? {} : finalDimensions),
  } as Invocation<DenoiseLatentsNodes>);

  copyDenoiseInputs(g, denoise, hrfDenoise, finalDimensions);

  if (denoise.type === 'denoise_latents') {
    assert(noise, 'SD1.5/SD2/SDXL high resolution fix graphs require a noise node');
    const classicHrfDenoise = hrfDenoise as Invocation<'denoise_latents'>;

    const hrfNoise = g.addNode({
      type: 'noise',
      id: getPrefixedId('hrf_noise'),
      use_cpu: noise.use_cpu,
      ...finalDimensions,
    });
    g.addEdge(seed, 'value', hrfNoise, 'seed');
    g.addEdge(hrfNoise, 'noise', classicHrfDenoise, 'noise');
  }

  g.deleteEdgesTo(l2i, ['latents']);
  g.addEdge(denoise, 'latents', resizeLatents, 'latents');
  g.addEdge(resizeLatents, 'latents', hrfDenoise, 'latents');
  g.addEdge(hrfDenoise, 'latents', l2i, 'latents');
  if (l2i.type === 'l2i') {
    g.updateNode(l2i, { tile_size: LATENT_HRF_L2I_TILE_SIZE, tiled: true });
  }

  g.upsertMetadata({
    width: params.dimensions.width,
    height: params.dimensions.height,
    hrf_enabled: true,
    hrf_method: 'latent',
    hrf_strength: params.hrfStrength,
    hrf_scale: params.hrfScale,
    hrf_latent_interpolation_mode: params.hrfLatentInterpolationMode,
  });

  return l2i;
};

const addUpscaleModelHighResFix = ({ g, state, denoise, l2i, noise, seed }: AddHighResFixArg): Invocation<'l2i'> => {
  const params = selectParamsSlice(state);
  const finalDimensions = getHighResFixFinalDimensions(state);
  const { denoising_start, denoising_end } = getHighResFixDenoisingStartAndEnd(state);

  assert(params.model?.base === 'sd-1' || params.model?.base === 'sdxl', 'Upscale model HRF supports SD1.5 and SDXL');
  assert(params.hrfUpscaleModel, 'Upscale model HRF requires a Spandrel upscale model');
  assert(params.hrfTileControlNetModel, 'Upscale model HRF requires a tile ControlNet model');
  assert(denoise.type === 'denoise_latents', 'Upscale model HRF requires classic SD denoise latents');
  assert(l2i.type === 'l2i', 'Upscale model HRF requires classic SD latents-to-image');
  assert(noise, 'Upscale model HRF requires a noise node');

  const intermediateL2i = g.addNode({
    ...l2i,
    id: getPrefixedId('hrf_intermediate_l2i'),
    is_intermediate: true,
    board: undefined,
    metadata: undefined,
  });
  copyInputEdges(g, l2i, intermediateL2i, SKIPPED_LATENT_TO_IMAGE_INPUT_FIELDS);

  const spandrelAutoscale = g.addNode({
    type: 'spandrel_image_to_image_autoscale',
    id: getPrefixedId('hrf_spandrel_autoscale'),
    image_to_image_model: params.hrfUpscaleModel,
    fit_to_multiple_of_8: true,
    scale: params.hrfScale,
    tile_size: params.hrfTileSize,
  });

  const unsharpMask = g.addNode({
    type: 'unsharp_mask',
    id: getPrefixedId('hrf_unsharp_2'),
    radius: 2,
    strength: 60,
  });

  const i2l = g.addNode({
    type: 'i2l',
    id: getPrefixedId('hrf_i2l'),
    fp32: l2i.fp32,
    tile_size: params.hrfTileSize,
    tiled: true,
  });
  copyInputEdges(g, l2i, i2l, SKIPPED_LATENT_TO_IMAGE_INPUT_FIELDS);
  g.updateNode(l2i, { tile_size: params.hrfTileSize, tiled: true });

  const hrfNoise = g.addNode({
    type: 'noise',
    id: getPrefixedId('hrf_noise'),
    use_cpu: noise.use_cpu,
  });

  const useClassicDenoise = hasUnsupportedTiledDenoiseInputs(g, denoise);
  const hrfSteps = params.hrfSteps ?? denoise.steps;
  const needsFreshHrfModelInputs = params.hrfModel !== null || params.hrfLoraMode !== 'reuse_generate';
  const hrfDenoise = useClassicDenoise
    ? g.addNode({
        ...denoise,
        id: getPrefixedId('hrf_denoise_latents'),
        steps: hrfSteps,
        denoising_start,
        denoising_end,
      })
    : g.addNode({
        type: 'tiled_multi_diffusion_denoise_latents',
        id: getPrefixedId('hrf_tiled_multidiffusion_denoise_latents'),
        tile_height: params.hrfTileSize,
        tile_width: params.hrfTileSize,
        tile_overlap: params.hrfTileOverlap,
        steps: hrfSteps,
        cfg_scale: denoise.cfg_scale,
        cfg_rescale_multiplier: denoise.cfg_rescale_multiplier,
        scheduler: denoise.scheduler,
        denoising_start,
        denoising_end,
      });

  copyDenoiseInputs(
    g,
    denoise,
    hrfDenoise,
    finalDimensions,
    useClassicDenoise ? undefined : TILED_DENOISE_INPUT_FIELDS,
    needsFreshHrfModelInputs ? FRESH_HRF_MODEL_INPUT_FIELDS : undefined
  );

  if (needsFreshHrfModelInputs) {
    if (params.hrfModel) {
      g.deleteEdgesTo(i2l, ['vae']);
      g.deleteEdgesTo(l2i, ['vae']);
    }

    const { vaeSource: hrfVaeSource } = addFreshHrfModelInputs(state, g, denoise, hrfDenoise, finalDimensions);
    if (params.hrfModel) {
      g.addEdge(hrfVaeSource, 'vae', i2l, 'vae');
      g.addEdge(hrfVaeSource, 'vae', l2i, 'vae');
    }
  }
  addTileControlNet(
    g,
    hrfDenoise,
    unsharpMask,
    params.hrfTileControlNetModel,
    params.hrfTileControlWeight,
    params.hrfTileControlEnd
  );

  g.deleteEdgesTo(l2i, ['latents']);
  g.addEdge(denoise, 'latents', intermediateL2i, 'latents');
  g.addEdge(intermediateL2i, 'image', spandrelAutoscale, 'image');
  g.addEdge(spandrelAutoscale, 'image', unsharpMask, 'image');
  g.addEdge(unsharpMask, 'image', i2l, 'image');
  g.addEdge(seed, 'value', hrfNoise, 'seed');
  g.addEdgeFromObj({
    source: { node_id: unsharpMask.id, field: 'width' },
    destination: { node_id: hrfNoise.id, field: 'width' },
  });
  g.addEdgeFromObj({
    source: { node_id: unsharpMask.id, field: 'height' },
    destination: { node_id: hrfNoise.id, field: 'height' },
  });
  g.addEdgeFromObj({
    source: { node_id: hrfNoise.id, field: 'noise' },
    destination: { node_id: hrfDenoise.id, field: 'noise' },
  });
  g.addEdgeFromObj({
    source: { node_id: i2l.id, field: 'latents' },
    destination: { node_id: hrfDenoise.id, field: 'latents' },
  });
  g.addEdgeFromObj({
    source: { node_id: hrfDenoise.id, field: 'latents' },
    destination: { node_id: l2i.id, field: 'latents' },
  });

  g.upsertMetadata({
    width: params.dimensions.width,
    height: params.dimensions.height,
    hrf_enabled: true,
    hrf_method: 'upscale_model',
    hrf_strength: params.hrfStrength,
    hrf_scale: params.hrfScale,
    hrf_upscale_model: params.hrfUpscaleModel,
    hrf_tile_controlnet_model: params.hrfTileControlNetModel,
    hrf_tile_control_weight: params.hrfTileControlWeight,
    hrf_tile_control_end: params.hrfTileControlEnd,
    hrf_tile_size: params.hrfTileSize,
    hrf_tile_overlap: params.hrfTileOverlap,
    hrf_steps: params.hrfSteps ?? undefined,
    hrf_model: params.hrfModel ?? undefined,
    hrf_lora_mode: params.hrfLoraMode,
    hrf_loras: getHrfLoRAMetadata(state),
  });

  return l2i;
};

export const addHighResFix = (arg: AddHighResFixArg): Invocation<LatentToImageNodes> => {
  const { g, state, generationMode, l2i } = arg;
  const params = selectParamsSlice(state);

  if (!shouldApplyHighResFix(state, generationMode)) {
    if (shouldWriteDisabledMetadata(state, generationMode)) {
      g.upsertMetadata({ hrf_enabled: false });
    }
    return l2i;
  }

  if (params.hrfMethod === 'upscale_model') {
    return addUpscaleModelHighResFix(arg);
  }

  return addLatentHighResFix(arg);
};
