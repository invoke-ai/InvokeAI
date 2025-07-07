import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { isChatGPT4oAspectRatioID, isChatGPT4oReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { type ImageField, zModelIdentifierField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields, selectOriginalAndScaledSizes } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { t } from 'i18next';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

export const buildChatGPT4oGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  if (generationMode !== 'txt2img' && generationMode !== 'img2img') {
    throw new UnsupportedGenerationModeError(t('toast.chatGPT4oIncompatibleGenerationMode'));
  }

  log.debug({ generationMode, manager: manager?.id }, 'Building ChatGPT 4o graph');

  const model = selectMainModelConfig(state);

  const refImages = selectRefImagesSlice(state);

  const { originalSize, scaledSize, aspectRatio } = selectOriginalAndScaledSizes(state);

  assert(model, 'No model selected');
  assert(model.base === 'chatgpt-4o', 'Selected model is not a ChatGPT 4o API model');

  const validRefImages = refImages.entities
    .filter((entity) => entity.isEnabled)
    .filter((entity) => isChatGPT4oReferenceImageConfig(entity.config))
    .filter((entity) => getGlobalReferenceImageWarnings(entity, model).length === 0)
    .toReversed(); // sends them in order they are displayed in the list

  let reference_images: ImageField[] | undefined = undefined;

  if (validRefImages.length > 0) {
    reference_images = [];
    for (const entity of validRefImages) {
      assert(entity.config.image, 'Image is required for reference image');
      reference_images.push({
        image_name: entity.config.image.image_name,
      });
    }
  }

  if (generationMode === 'txt2img') {
    assert(isChatGPT4oAspectRatioID(aspectRatio.id), 'ChatGPT 4o does not support this aspect ratio');

    const g = new Graph(getPrefixedId('chatgpt_4o_txt2img_graph'));
    const positivePrompt = g.addNode({
      id: getPrefixedId('positive_prompt'),
      type: 'string',
    });
    const gptImage = g.addNode({
      // @ts-expect-error: These nodes are not available in the OSS application
      type: 'chatgpt_4o_generate_image',
      model: zModelIdentifierField.parse(model),
      aspect_ratio: aspectRatio.id,
      reference_images,
      ...selectCanvasOutputFields(state),
    });

    g.addEdge(
      positivePrompt,
      'value',
      gptImage,
      // @ts-expect-error: These nodes are not available in the OSS application
      'positive_prompt'
    );
    g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');
    g.upsertMetadata({
      model: Graph.getModelMetadataField(model),
      width: originalSize.width,
      height: originalSize.height,
    });
    return {
      g,
      positivePrompt,
    };
  }

  if (generationMode === 'img2img') {
    assert(manager !== null);
    const adapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
    const { image_name } = await manager.compositor.getCompositeImageDTO(adapters, bbox.rect, {
      is_intermediate: true,
      silent: true,
    });
    const g = new Graph(getPrefixedId('chatgpt_4o_img2img_graph'));
    const positivePrompt = g.addNode({
      id: getPrefixedId('positive_prompt'),
      type: 'string',
    });
    const gptImage = g.addNode({
      // @ts-expect-error: These nodes are not available in the OSS application
      type: 'chatgpt_4o_edit_image',
      model: zModelIdentifierField.parse(model),
      aspect_ratio: bbox.aspectRatio.id,
      base_image: { image_name },
      reference_images,
      ...selectCanvasOutputFields(state),
    });

    g.addEdge(
      positivePrompt,
      'value',
      gptImage,
      // @ts-expect-error: These nodes are not available in the OSS application
      'positive_prompt'
    );
    g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');
    g.upsertMetadata({
      model: Graph.getModelMetadataField(model),
      width: bbox.rect.width,
      height: bbox.rect.height,
    });
    return {
      g,
      positivePrompt,
    };
  }

  assert<Equals<typeof generationMode, never>>(false, 'Invalid generation mode for ChatGPT ');
};
