import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { isFluxKontextAspectRatioID, isFluxKontextReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { zImageField, zModelIdentifierField } from 'features/nodes/types/common';
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

export const buildFluxKontextGraph = (arg: GraphBuilderArg): GraphBuilderReturn => {
  const { generationMode, state, manager } = arg;

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'flux-kontext', 'Selected model is not a FLUX Kontext API model');

  if (generationMode !== 'txt2img') {
    throw new UnsupportedGenerationModeError(t('toast.fluxKontextIncompatibleGenerationMode'));
  }

  log.debug({ generationMode, manager: manager?.id }, 'Building FLUX Kontext graph');

  const { originalSize, aspectRatio } = getOriginalAndScaledSizesForTextToImage(state);
  assert(isFluxKontextAspectRatioID(aspectRatio.id), 'FLUX Kontext does not support this aspect ratio');

  const refImages = selectRefImagesSlice(state);

  const validRefImages = refImages.entities
    .filter((entity) => entity.isEnabled)
    .filter((entity) => isFluxKontextReferenceImageConfig(entity.config))
    .filter((entity) => getGlobalReferenceImageWarnings(entity, model).length === 0);

  const g = new Graph(getPrefixedId('flux_kontext_txt2img_graph'));
  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });

  let fluxKontextImage;

  if (validRefImages.length > 0) {
    if (validRefImages.length === 1) {
      // Single reference image - use it directly
      const firstImage = validRefImages[0]?.config.image;
      assert(firstImage, 'First image should exist when validRefImages.length > 0');

      fluxKontextImage = g.addNode({
        // @ts-expect-error: These nodes are not available in the OSS application
        type: 'flux_kontext_edit_image',
        model: zModelIdentifierField.parse(model),
        aspect_ratio: aspectRatio.id,
        prompt_upsampling: true,
        input_image: {
          image_name: firstImage.image_name,
        },
        ...selectCanvasOutputFields(state),
      });
    } else {
      // Multiple reference images - use concatenation
      const kontextConcatenator = g.addNode({
        id: getPrefixedId('flux_kontext_image_prep'),
        type: 'flux_kontext_image_prep',
        images: validRefImages.map(({ config }) => zImageField.parse(config.image)),
      });

      fluxKontextImage = g.addNode({
        // @ts-expect-error: These nodes are not available in the OSS application
        type: 'flux_kontext_edit_image',
        model: zModelIdentifierField.parse(model),
        aspect_ratio: aspectRatio.id,
        prompt_upsampling: true,
        input_image: {
          image_name: kontextConcatenator.id,
        },
        ...selectCanvasOutputFields(state),
      });
    }
  } else {
    fluxKontextImage = g.addNode({
      // @ts-expect-error: These nodes are not available in the OSS application
      type: 'flux_kontext_generate_image',
      model: zModelIdentifierField.parse(model),
      aspect_ratio: aspectRatio.id,
      prompt_upsampling: true,
      ...selectCanvasOutputFields(state),
    });
  }

  g.addEdge(
    positivePrompt,
    'value',
    fluxKontextImage,
    // @ts-expect-error: These nodes are not available in the OSS application
    'positive_prompt'
  );
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  g.upsertMetadata({
    model: Graph.getModelMetadataField(model),
    width: originalSize.width,
    height: originalSize.height,
  });

  if (validRefImages.length > 0) {
    g.upsertMetadata({ ref_images: [validRefImages] }, 'merge');
  }

  g.setMetadataReceivingNode(fluxKontextImage);

  return {
    g,
    positivePrompt,
  };
};
