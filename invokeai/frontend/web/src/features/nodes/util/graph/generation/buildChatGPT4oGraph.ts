import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import { isChatGPT4oAspectRatioID, isChatGPT4oReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { type ImageField, zModelIdentifierField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getOriginalAndScaledSizesForOtherModes,
  getOriginalAndScaledSizesForTextToImage,
  selectCanvasOutputFields,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
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
    const { originalSize, aspectRatio } = getOriginalAndScaledSizesForTextToImage(state);
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
  } else if (generationMode === 'img2img') {
    const { aspectRatio, rect } = getOriginalAndScaledSizesForOtherModes(state);
    assert(isChatGPT4oAspectRatioID(aspectRatio.id), 'ChatGPT 4o does not support this aspect ratio');

    assert(manager !== null);
    const adapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
    const { image_name } = await manager.compositor.getCompositeImageDTO(adapters, rect, {
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
      aspect_ratio: aspectRatio.id,
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
      width: rect.width,
      height: rect.height,
    });

    if (selectActiveTab(state) === 'canvas') {
      g.upsertMetadata(selectCanvasMetadata(state));
    }

    g.setMetadataReceivingNode(gptImage);

    return {
      g,
      positivePrompt,
    };
  }

  assert<Equals<typeof generationMode, never>>(false, 'Invalid generation mode for ChatGPT ');
};
