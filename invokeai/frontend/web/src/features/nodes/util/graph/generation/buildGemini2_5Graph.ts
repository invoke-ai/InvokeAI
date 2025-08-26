import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { isGemini2_5AspectRatioID, isGemini2_5ReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import type { ImageField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getOriginalAndScaledSizesForTextToImage,
  selectCanvasOutputFields,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { t } from 'i18next';
import { assert } from 'tsafe';

const log = logger('system');

export const buildGemini2_5Graph = (arg: GraphBuilderArg): GraphBuilderReturn => {
  const { generationMode, state, manager } = arg;

  if (generationMode !== 'txt2img') {
    throw new UnsupportedGenerationModeError(t('toast.chatGPT4oIncompatibleGenerationMode'));
  }

  log.debug({ generationMode, manager: manager?.id }, 'Building Gemini 2.5 graph');

  const model = selectMainModelConfig(state);

  const refImages = selectRefImagesSlice(state);

  assert(model, 'No model selected');
  assert(model.base === 'gemini-2.5', 'Selected model is not a Gemini 2.5 API model');

  const validRefImages = refImages.entities
    .filter((entity) => entity.isEnabled)
    .filter((entity) => isGemini2_5ReferenceImageConfig(entity.config))
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

  const { originalSize, aspectRatio } = getOriginalAndScaledSizesForTextToImage(state);
  assert(isGemini2_5AspectRatioID(aspectRatio.id), 'Gemini 2.5 does not support this aspect ratio');

  const g = new Graph(getPrefixedId('gemini_2_5_txt2img_graph'));
  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const geminiImage = g.addNode({
    // @ts-expect-error: These nodes are not available in the OSS application
    type: 'google_gemini_generate_image',
    reference_images,
    ...selectCanvasOutputFields(state),
  });

  g.addEdge(
    positivePrompt,
    'value',
    geminiImage,
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
};
