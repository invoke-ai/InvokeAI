import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasMetadata, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { isImagenAspectRatioID } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields, selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { t } from 'i18next';
import { assert } from 'tsafe';

const log = logger('system');

export const buildImagen3Graph = (arg: GraphBuilderArg): GraphBuilderReturn => {
  const { generationMode, state, manager } = arg;

  if (generationMode !== 'txt2img') {
    throw new UnsupportedGenerationModeError(t('toast.imagenIncompatibleGenerationMode', { model: 'Imagen3' }));
  }

  log.debug({ generationMode, manager: manager?.id }, 'Building Imagen3 graph');

  const canvas = selectCanvasSlice(state);

  const { bbox } = canvas;
  const { positivePrompt, negativePrompt } = selectPresetModifiedPrompts(state);
  const model = selectMainModelConfig(state);

  assert(model, 'No model found for Imagen3 graph');
  assert(model.base === 'imagen3', 'Imagen3 graph requires Imagen3 model');
  assert(isImagenAspectRatioID(bbox.aspectRatio.id), 'Imagen3 does not support this aspect ratio');
  assert(positivePrompt.length > 0, 'Imagen3 requires positive prompt to have at least one character');

  const g = new Graph(getPrefixedId('imagen3_txt2img_graph'));
  const imagen3 = g.addNode({
    // @ts-expect-error: These nodes are not available in the OSS application
    type: 'google_imagen3_generate_image',
    model: zModelIdentifierField.parse(model),
    positive_prompt: positivePrompt,
    negative_prompt: negativePrompt,
    aspect_ratio: bbox.aspectRatio.id,
    // When enhance_prompt is true, Imagen3 will return a new image every time, ignoring the seed.
    enhance_prompt: true,
    ...selectCanvasOutputFields(state),
  });
  g.upsertMetadata({
    positive_prompt: positivePrompt,
    negative_prompt: negativePrompt,
    width: bbox.rect.width,
    height: bbox.rect.height,
    model: Graph.getModelMetadataField(model),
    ...selectCanvasMetadata(state),
  });

  return {
    g,
    seedFieldIdentifier: { nodeId: imagen3.id, fieldName: 'seed' },
    positivePromptFieldIdentifier: { nodeId: imagen3.id, fieldName: 'positive_prompt' },
  };
};
