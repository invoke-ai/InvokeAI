import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { isFluxKontextReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import type { ImageField } from 'features/nodes/types/common';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields, selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { t } from 'i18next';
import { assert } from 'tsafe';

const log = logger('system');

export const buildFluxKontextGraph = (arg: GraphBuilderArg): GraphBuilderReturn => {
  const { generationMode, state, manager } = arg;

  if (generationMode !== 'txt2img') {
    throw new UnsupportedGenerationModeError(t('toast.imagenIncompatibleGenerationMode', { model: 'FLUX Kontext' }));
  }

  log.debug({ generationMode, manager: manager?.id }, 'Building FLUX Kontext graph');

  const model = selectMainModelConfig(state);

  const canvas = selectCanvasSlice(state);
  const refImages = selectRefImagesSlice(state);

  const { bbox } = canvas;
  const { positivePrompt } = selectPresetModifiedPrompts(state);

  assert(model, 'No model found in state');
  assert(model.base === 'flux-kontext', 'Model is not a Flux Kontext model');

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
  const fluxKontextImage = g.addNode({
    // @ts-expect-error: These nodes are not available in the OSS application
    type: input_image ? 'flux_kontext_edit_image' : 'flux_kontext_generate_image',
    model: zModelIdentifierField.parse(model),
    positive_prompt: positivePrompt,
    aspect_ratio: bbox.aspectRatio.id,
    input_image,
    prompt_upsampling: true,
    ...selectCanvasOutputFields(state),
  });
  g.upsertMetadata({
    positive_prompt: positivePrompt,
    model: Graph.getModelMetadataField(model),
    width: bbox.rect.width,
    height: bbox.rect.height,
  });
  return {
    g,
    positivePromptFieldIdentifier: { nodeId: fluxKontextImage.id, fieldName: 'positive_prompt' },
  };
};
