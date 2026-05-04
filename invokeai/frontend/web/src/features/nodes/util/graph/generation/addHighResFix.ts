import type { RootState } from 'app/store/store';
import { roundDownToMultiple } from 'common/util/roundDownToMultiple';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import type { GenerationMode } from 'features/controlLayers/store/types';
import type { BaseModelType } from 'features/nodes/types/common';
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

const SKIPPED_DENOISE_INPUT_FIELDS = new Set(['latents', 'noise']);
const TILED_DENOISE_INPUT_FIELDS = new Set(['positive_conditioning', 'negative_conditioning', 'unet']);
const SKIPPED_LATENT_TO_IMAGE_INPUT_FIELDS = new Set(['latents', 'metadata']);

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
    model.base !== 'external' &&
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
  finalDimensions: { width: number; height: number }
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
    return null;
  }

  const collect = g.addNode({
    type: 'collect',
    id: getPrefixedId('hrf_sdxl_conditioning_collect'),
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
  allowedInputFields?: Set<string>
) => {
  for (const edge of g.getEdgesTo(from)) {
    if (SKIPPED_DENOISE_INPUT_FIELDS.has(edge.destination.field)) {
      continue;
    }
    if (allowedInputFields && !allowedInputFields.has(edge.destination.field)) {
      continue;
    }

    const finalSizeConditioning = ['positive_conditioning', 'negative_conditioning'].includes(edge.destination.field)
      ? cloneSDXLConditioningForFinalDimensions(g, edge.source.node_id, finalDimensions)
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
    return !SKIPPED_DENOISE_INPUT_FIELDS.has(field) && !TILED_DENOISE_INPUT_FIELDS.has(field);
  });
};

const addTileControlNets = (
  g: Graph,
  hrfDenoise: AnyInvocation,
  imageSource: Invocation<'unsharp_mask'>,
  tileControlNetModel: NonNullable<ReturnType<typeof selectParamsSlice>['hrfTileControlNetModel']>,
  structure: number
) => {
  const controlNet1 = g.addNode({
    id: getPrefixedId('hrf_controlnet_1'),
    type: 'controlnet',
    control_model: tileControlNetModel,
    control_mode: 'balanced',
    resize_mode: 'just_resize',
    control_weight: (structure + 10) * 0.0325 + 0.3,
    begin_step_percent: 0,
    end_step_percent: (structure + 10) * 0.025 + 0.3,
  });

  const controlNet2 = g.addNode({
    id: getPrefixedId('hrf_controlnet_2'),
    type: 'controlnet',
    control_model: tileControlNetModel,
    control_mode: 'balanced',
    resize_mode: 'just_resize',
    control_weight: ((structure + 10) * 0.0325 + 0.15) * 0.45,
    begin_step_percent: (structure + 10) * 0.025 + 0.3,
    end_step_percent: 0.85,
  });

  const controlNetCollector = g.addNode({
    type: 'collect',
    id: getPrefixedId('hrf_controlnet_collector'),
  });

  g.addEdge(imageSource, 'image', controlNet1, 'image');
  g.addEdge(imageSource, 'image', controlNet2, 'image');
  g.addEdge(controlNet1, 'control', controlNetCollector, 'item');
  g.addEdge(controlNet2, 'control', controlNetCollector, 'item');
  g.addEdgeFromObj({
    source: { node_id: controlNetCollector.id, field: 'collection' },
    destination: { node_id: hrfDenoise.id, field: 'control' },
  });
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

  g.upsertMetadata({
    width: finalDimensions.width,
    height: finalDimensions.height,
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
  const hrfDenoise = useClassicDenoise
    ? g.addNode({
        ...denoise,
        id: getPrefixedId('hrf_denoise_latents'),
        denoising_start,
        denoising_end,
      })
    : g.addNode({
        type: 'tiled_multi_diffusion_denoise_latents',
        id: getPrefixedId('hrf_tiled_multidiffusion_denoise_latents'),
        tile_height: params.hrfTileSize,
        tile_width: params.hrfTileSize,
        tile_overlap: params.hrfTileOverlap,
        steps: denoise.steps,
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
    useClassicDenoise ? undefined : TILED_DENOISE_INPUT_FIELDS
  );
  addTileControlNets(g, hrfDenoise, unsharpMask, params.hrfTileControlNetModel, params.hrfStructure);

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
    width: finalDimensions.width,
    height: finalDimensions.height,
    hrf_enabled: true,
    hrf_method: 'upscale_model',
    hrf_strength: params.hrfStrength,
    hrf_scale: params.hrfScale,
    hrf_upscale_model: params.hrfUpscaleModel,
    hrf_tile_controlnet_model: params.hrfTileControlNetModel,
    hrf_structure: params.hrfStructure,
    hrf_tile_size: params.hrfTileSize,
    hrf_tile_overlap: params.hrfTileOverlap,
  });
  g.addEdgeToMetadata(spandrelAutoscale, 'width', 'width');
  g.addEdgeToMetadata(spandrelAutoscale, 'height', 'height');

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
