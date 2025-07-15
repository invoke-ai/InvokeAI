import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { isFluxKontextAspectRatioID, isFluxKontextReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import type { ImageField } from 'features/nodes/types/common';
import { zModelIdentifierField } from 'features/nodes/types/common';
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
    .filter((entity) => getGlobalReferenceImageWarnings(entity, model).length === 0)
    .toReversed(); // sends them in order they are displayed in the list

  let input_image: ImageField | undefined = undefined;

  if (validRefImages[0]) {
    assert(validRefImages.length === 1, 'Flux Kontext can have at most one reference image');

    assert(validRefImages[0].config.image, 'Image is required for reference image');
    input_image = {
      image_name: validRefImages[0].config.image.image_name,
    };
  }

  const g = new Graph(getPrefixedId('flux_kontext_txt2img_graph'));
  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const fluxKontextImage = g.addNode({
    // @ts-expect-error: These nodes are not available in the OSS application
    type: input_image ? 'flux_kontext_edit_image' : 'flux_kontext_generate_image',
    model: zModelIdentifierField.parse(model),
    aspect_ratio: aspectRatio.id,
    input_image,
    prompt_upsampling: true,
    ...selectCanvasOutputFields(state),
  });

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
  return {
    g,
    positivePrompt,
  };
};
