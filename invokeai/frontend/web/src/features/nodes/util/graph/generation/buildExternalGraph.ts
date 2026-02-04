import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { zImageField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getOriginalAndScaledSizesForOtherModes,
  getOriginalAndScaledSizesForTextToImage,
  selectCanvasOutputFields,
  selectPresetModifiedPrompts,
} from 'features/nodes/util/graph/graphBuilderUtils';
import {
  type GraphBuilderArg,
  type GraphBuilderReturn,
  UnsupportedGenerationModeError,
} from 'features/nodes/util/graph/types';
import { type Invocation, isExternalApiModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

export const buildExternalGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  const model = selectModelConfig(state);
  assert(model, 'No model selected');
  assert(isExternalApiModelConfig(model), 'Selected model is not an external API model');

  const requestedMode = generationMode === 'outpaint' ? 'inpaint' : generationMode;
  if (!model.capabilities.modes.includes(requestedMode)) {
    throw new UnsupportedGenerationModeError(`${model.name} does not support ${requestedMode} mode`);
  }

  const params = selectParamsSlice(state);
  const refImages = selectRefImagesSlice(state);
  const prompts = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('external_graph'));

  const seed = model.capabilities.supports_seed
    ? g.addNode({
        id: getPrefixedId('seed'),
        type: 'integer',
      })
    : null;

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });

  const externalNode = g.addNode({
    id: getPrefixedId('external_image_generation'),
    type: 'external_image_generation',
    model,
    mode: requestedMode,
    negative_prompt: model.capabilities.supports_negative_prompt ? prompts.negative : null,
    steps: params.steps,
    guidance: model.capabilities.supports_guidance ? params.guidance : null,
    num_images: 1,
  });

  if (seed) {
    g.addEdge(seed, 'value', externalNode, 'seed');
  }
  g.addEdge(positivePrompt, 'value', externalNode, 'prompt');

  if (model.capabilities.supports_reference_images) {
    const referenceImages = refImages.entities
      .filter((entity) => entity.isEnabled)
      .map((entity) => entity.config)
      .filter((config) => config.image)
      .map((config) => zImageField.parse(config.image?.crop?.image ?? config.image?.original.image));

    const referenceWeights = refImages.entities
      .filter((entity) => entity.isEnabled)
      .map((entity) => entity.config)
      .filter((config) => config.image)
      .map((config) => (config.type === 'ip_adapter' ? config.weight : null));

    if (referenceImages.length > 0) {
      externalNode.reference_images = referenceImages;
      if (referenceWeights.every((weight): weight is number => weight !== null)) {
        externalNode.reference_image_weights = referenceWeights;
      }
    }
  }

  if (generationMode === 'txt2img') {
    const { scaledSize } = getOriginalAndScaledSizesForTextToImage(state);
    externalNode.width = scaledSize.width;
    externalNode.height = scaledSize.height;
  } else {
    assert(manager, 'Canvas manager is required for img2img/inpaint');
    const canvasSettings = selectCanvasSettingsSlice(state);
    const { scaledSize, rect } = getOriginalAndScaledSizesForOtherModes(state);
    externalNode.width = scaledSize.width;
    externalNode.height = scaledSize.height;

    const rasterAdapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
    const initImage = await manager.compositor.getCompositeImageDTO(rasterAdapters, rect, {
      is_intermediate: true,
      silent: true,
    });
    externalNode.init_image = { image_name: initImage.image_name };

    if (generationMode === 'inpaint' || generationMode === 'outpaint') {
      const inpaintMaskAdapters = manager.compositor.getVisibleAdaptersOfType('inpaint_mask');
      const maskImage = await manager.compositor.getGrayscaleMaskCompositeImageDTO(
        inpaintMaskAdapters,
        rect,
        'denoiseLimit',
        canvasSettings.preserveMask,
        {
          is_intermediate: true,
          silent: true,
        }
      );
      externalNode.mask_image = { image_name: maskImage.image_name };
    }
  }

  g.updateNode(externalNode, selectCanvasOutputFields(state));

  return {
    g,
    seed: seed ?? undefined,
    positivePrompt: positivePrompt as Invocation<'string'>,
  };
};
