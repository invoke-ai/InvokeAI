import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getBoardField } from 'features/nodes/util/graph/graphBuilderUtils';
import { selectSimpleGenerationSlice } from 'features/simpleGeneration/store/slice';
import { ASPECT_RATIO_MAP } from 'features/simpleGeneration/util/aspectRatioToDimensions';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';

const EMPTY_ENTITY_STATE = {
  ids: [],
  entities: {},
};

export const getFLUXModels = (modelConfigs: ReturnType<typeof selectModelConfigsQuery>) => {
  const allModelConfigs = modelConfigsAdapterSelectors.selectAll(modelConfigs?.data ?? EMPTY_ENTITY_STATE);

  /**
   * The sources are taken from `invokeai/backend/model_manager/starter_models.py`
   *
   * Note: The sources for HF Repo models are subtly in the python file vs the actual model config we get from the
   * HTTP query response.
   *
   * In python, a double colon `::` separates the HF Repo ID from the folder path within the repo. But the model
   * configs we get from the query use a single colon `:`.
   *
   * For example:
   * Python: InvokeAI/t5-v1_1-xxl::bnb_llm_int8
   * Query : InvokeAI/t5-v1_1-xxl:bnb_llm_int8
   *
   * I'm not sure if it's always been like this, but to be safe, we check for both formats below.
   */

  const flux = allModelConfigs.find(
    ({ type, base, source }) =>
      type === 'main' &&
      base === 'flux' &&
      (source === 'InvokeAI/flux_dev:transformer/bnb_nf4/flux1-dev-bnb_nf4.safetensors' ||
        source === 'InvokeAI/flux_dev::transformer/bnb_nf4/flux1-dev-bnb_nf4.safetensors')
  );

  const t5Encoder = allModelConfigs.find(
    ({ type, base, source }) =>
      type === 't5_encoder' &&
      base === 'any' &&
      (source === 'InvokeAI/t5-v1_1-xxl:bnb_llm_int8' || source === 'InvokeAI/t5-v1_1-xxl::bnb_llm_int8')
  );

  const clipEmbed = allModelConfigs.find(
    ({ type, base, source }) =>
      type === 'clip_embed' &&
      base === 'any' &&
      (source === 'InvokeAI/clip-vit-large-patch14-text-encoder:bfloat16' ||
        source === 'InvokeAI/clip-vit-large-patch14-text-encoder::bfloat16')
  );

  const clipVision = allModelConfigs.find(
    ({ type, base, source }) => type === 'clip_vision' && base === 'any' && source === 'InvokeAI/clip-vit-large-patch14'
  );

  const vae = allModelConfigs.find(
    ({ type, base, source }) =>
      type === 'vae' &&
      base === 'flux' &&
      (source === 'black-forest-labs/FLUX.1-schnell:ae.safetensors' ||
        source === 'black-forest-labs/FLUX.1-schnell::ae.safetensors')
  );

  const ipAdapter = allModelConfigs.find(
    ({ type, base, source }) =>
      type === 'ip_adapter' &&
      base === 'flux' &&
      source === 'https://huggingface.co/XLabs-AI/flux-ip-adapter-v2/resolve/main/ip_adapter.safetensors'
  );

  return {
    flux,
    t5Encoder,
    clipEmbed,
    clipVision,
    vae,
    ipAdapter,
  };
};

export const getSD1Models = (modelConfigs: ReturnType<typeof selectModelConfigsQuery>) => {
  const allModelConfigs = modelConfigsAdapterSelectors.selectAll(modelConfigs?.data ?? EMPTY_ENTITY_STATE);

  /**
   * The sources are taken from `invokeai/backend/model_manager/starter_models.py`
   *
   * Note: The sources for HF Repo models are subtly in the python file vs the actual model config we get from the
   * HTTP query response.
   *
   * In python, a double colon `::` separates the HF Repo ID from the folder path within the repo. But the model
   * configs we get from the query use a single colon `:`.
   *
   * For example:
   * Python: InvokeAI/t5-v1_1-xxl::bnb_llm_int8
   * Query : InvokeAI/t5-v1_1-xxl:bnb_llm_int8
   *
   * I'm not sure if it's always been like this, but to be safe, we check for both formats below.
   */

  const main = allModelConfigs.find(
    ({ type, base, source }) =>
      type === 'main' &&
      base === 'sd-1' &&
      source === 'https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v5.safetensors'
  );

  return {
    main,
  };
};

export const getSDXLModels = (modelConfigs: ReturnType<typeof selectModelConfigsQuery>) => {
  const allModelConfigs = modelConfigsAdapterSelectors.selectAll(modelConfigs?.data ?? EMPTY_ENTITY_STATE);

  /**
   * The sources are taken from `invokeai/backend/model_manager/starter_models.py`
   *
   * Note: The sources for HF Repo models are subtly in the python file vs the actual model config we get from the
   * HTTP query response.
   *
   * In python, a double colon `::` separates the HF Repo ID from the folder path within the repo. But the model
   * configs we get from the query use a single colon `:`.
   *
   * For example:
   * Python: InvokeAI/t5-v1_1-xxl::bnb_llm_int8
   * Query : InvokeAI/t5-v1_1-xxl:bnb_llm_int8
   *
   * I'm not sure if it's always been like this, but to be safe, we check for both formats below.
   */

  const main = allModelConfigs.find(
    ({ type, base, source }) => type === 'main' && base === 'sdxl' && source === 'RunDiffusion/Juggernaut-XL-v9'
  );

  const vae = allModelConfigs.find(
    ({ type, base, source }) => type === 'vae' && base === 'sdxl' && source === 'madebyollin/sdxl-vae-fp16-fix'
  );

  return {
    main,
    vae,
  };
};

const buildSimpleSD1Graph = (state: RootState) => {
  const { positivePrompt, aspectRatio } = selectSimpleGenerationSlice(state);

  const { main } = getSD1Models(selectModelConfigsQuery(state));
  const g = new Graph(getPrefixedId('simple_sd1'));

  const dimensions = ASPECT_RATIO_MAP['sd-1'][aspectRatio];

  const modelLoader = g.addNode({
    type: 'main_model_loader',
    id: getPrefixedId('main_model_loader'),
    model: zModelIdentifierField.parse(main),
  });
  const posCond = g.addNode({
    type: 'compel',
    id: getPrefixedId('compel_prompt_pos'),
    prompt: positivePrompt,
  });
  const negCond = g.addNode({
    type: 'compel',
    id: getPrefixedId('compel_prompt_neg'),
    prompt: '',
  });
  const seed = g.addNode({
    type: 'rand_int',
    id: getPrefixedId('rand_int'),
    low: NUMPY_RAND_MIN,
    high: NUMPY_RAND_MAX,
    use_cache: false,
  });
  const noise = g.addNode({
    type: 'noise',
    id: getPrefixedId('noise'),
    width: dimensions.width,
    height: dimensions.height,
  });
  const denoise = g.addNode({
    type: 'denoise_latents',
    id: getPrefixedId('denoise_latents'),
    steps: 30,
  });
  const l2i = g.addNode({
    type: 'l2i',
    id: getPrefixedId('l2i'),
    is_intermediate: false,
    board: getBoardField(state),
  });

  g.addEdge(modelLoader, 'clip', posCond, 'clip');
  g.addEdge(modelLoader, 'clip', negCond, 'clip');

  g.addEdge(seed, 'value', noise, 'seed');

  g.addEdge(noise, 'noise', denoise, 'noise');
  g.addEdge(modelLoader, 'unet', denoise, 'unet');
  g.addEdge(posCond, 'conditioning', denoise, 'positive_conditioning');
  g.addEdge(negCond, 'conditioning', denoise, 'negative_conditioning');

  g.addEdge(modelLoader, 'vae', l2i, 'vae');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  return g;
};

const buildSimpleSDXLGraph = (state: RootState) => {
  const { positivePrompt, aspectRatio } = selectSimpleGenerationSlice(state);

  const { main, vae } = getSDXLModels(selectModelConfigsQuery(state));
  const g = new Graph(getPrefixedId('simple_sdxl'));

  const dimensions = ASPECT_RATIO_MAP['sdxl'][aspectRatio];

  const modelLoader = g.addNode({
    type: 'sdxl_model_loader',
    id: getPrefixedId('sdxl_model_loader'),
    model: zModelIdentifierField.parse(main),
  });
  const vaeLoader = g.addNode({
    type: 'vae_loader',
    id: getPrefixedId('vae_loader'),
    vae_model: zModelIdentifierField.parse(vae),
  });
  const posCond = g.addNode({
    type: 'sdxl_compel_prompt',
    id: getPrefixedId('sdxl_compel_prompt_pos'),
    prompt: positivePrompt,
  });
  const negCond = g.addNode({
    type: 'sdxl_compel_prompt',
    id: getPrefixedId('sdxl_compel_prompt_neg'),
    prompt: '',
  });
  const seed = g.addNode({
    type: 'rand_int',
    id: getPrefixedId('rand_int'),
    low: NUMPY_RAND_MIN,
    high: NUMPY_RAND_MAX,
    use_cache: false,
  });
  const noise = g.addNode({
    type: 'noise',
    id: getPrefixedId('noise'),
    width: dimensions.width,
    height: dimensions.height,
  });
  const denoise = g.addNode({
    type: 'denoise_latents',
    id: getPrefixedId('denoise_latents'),
    steps: 30,
  });
  const l2i = g.addNode({
    type: 'l2i',
    id: getPrefixedId('l2i'),
    is_intermediate: false,
    board: getBoardField(state),
  });

  g.addEdge(modelLoader, 'clip', posCond, 'clip');
  g.addEdge(modelLoader, 'clip2', posCond, 'clip2');

  g.addEdge(modelLoader, 'clip', negCond, 'clip');
  g.addEdge(modelLoader, 'clip2', negCond, 'clip2');

  g.addEdge(seed, 'value', noise, 'seed');

  g.addEdge(noise, 'noise', denoise, 'noise');
  g.addEdge(modelLoader, 'unet', denoise, 'unet');
  g.addEdge(posCond, 'conditioning', denoise, 'positive_conditioning');
  g.addEdge(negCond, 'conditioning', denoise, 'negative_conditioning');

  g.addEdge(vaeLoader, 'vae', l2i, 'vae');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  return g;
};

const buildSimpleFLUXGraph = (state: RootState) => {
  const { positivePrompt, aspectRatio } = selectSimpleGenerationSlice(state);

  const { flux, t5Encoder, clipEmbed, vae } = getFLUXModels(selectModelConfigsQuery(state));
  const g = new Graph(getPrefixedId('simple_flux'));

  const dimensions = ASPECT_RATIO_MAP['flux'][aspectRatio];

  const modelLoader = g.addNode({
    type: 'flux_model_loader',
    id: getPrefixedId('flux_model_loader'),
    model: zModelIdentifierField.parse(flux),
    t5_encoder_model: zModelIdentifierField.parse(t5Encoder),
    clip_embed_model: zModelIdentifierField.parse(clipEmbed),
    vae_model: zModelIdentifierField.parse(vae),
  });
  const posCond = g.addNode({
    type: 'flux_text_encoder',
    id: getPrefixedId('flux_text_encoder'),
    prompt: positivePrompt,
  });
  const seed = g.addNode({
    type: 'rand_int',
    id: getPrefixedId('rand_int'),
    low: NUMPY_RAND_MIN,
    high: NUMPY_RAND_MAX,
    use_cache: false,
  });
  const denoise = g.addNode({
    type: 'flux_denoise',
    id: getPrefixedId('flux_denoise'),
    guidance: 4.0,
    num_steps: 30,
    denoising_start: 0,
    denoising_end: 1,
    width: dimensions.width,
    height: dimensions.height,
  });
  const l2i = g.addNode({
    type: 'flux_vae_decode',
    id: getPrefixedId('flux_vae_decode'),
    is_intermediate: false,
    board: getBoardField(state),
  });

  g.addEdge(modelLoader, 't5_encoder', posCond, 't5_encoder');
  g.addEdge(modelLoader, 'max_seq_len', posCond, 't5_max_seq_len');
  g.addEdge(modelLoader, 'clip', posCond, 'clip');

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'vae', denoise, 'controlnet_vae');
  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(posCond, 'conditioning', denoise, 'positive_text_conditioning');

  g.addEdge(modelLoader, 'vae', l2i, 'vae');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  return g;
};

export const buildSimpleGraph = (state: RootState): Graph => {
  const { model } = selectSimpleGenerationSlice(state);

  if (model === 'flux') {
    return buildSimpleFLUXGraph(state);
  }

  if (model === 'sdxl') {
    return buildSimpleSDXLGraph(state);
  }

  if (model === 'sd-1') {
    return buildSimpleSD1Graph(state);
  }
};
