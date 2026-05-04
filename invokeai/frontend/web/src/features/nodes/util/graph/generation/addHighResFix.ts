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
import type { AnyInvocationInputField, AnyInvocationOutputField, Invocation } from 'services/api/types';
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
  from: Invocation<DenoiseLatentsNodes>,
  to: Invocation<DenoiseLatentsNodes>,
  finalDimensions: { width: number; height: number }
) => {
  for (const edge of g.getEdgesTo(from)) {
    if (SKIPPED_DENOISE_INPUT_FIELDS.has(edge.destination.field)) {
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

export const addHighResFix = ({
  g,
  state,
  generationMode,
  denoise,
  l2i,
  noise,
  seed,
}: AddHighResFixArg): Invocation<LatentToImageNodes> => {
  const params = selectParamsSlice(state);

  if (!shouldApplyHighResFix(state, generationMode)) {
    if (shouldWriteDisabledMetadata(state, generationMode)) {
      g.upsertMetadata({ hrf_enabled: false });
    }
    return l2i;
  }

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
