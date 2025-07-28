import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import { isImagenAspectRatioID } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getOriginalAndScaledSizesForTextToImage,
  selectCanvasOutputFields,
  selectPresetModifiedPrompts,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { t } from 'i18next';
import { assert } from 'tsafe';

const log = logger('system');

export const buildImagen3Graph = (arg: GraphBuilderArg): GraphBuilderReturn => {
  const { generationMode, state, manager } = arg;
  log.debug({ generationMode, manager: manager?.id }, 'Building Imagen3 graph');

  const model = selectMainModelConfig(state);

  assert(model, 'No model selected');
  assert(model.base === 'imagen3', 'Selected model is not an Imagen3 API model');

  if (generationMode !== 'txt2img') {
    throw new UnsupportedGenerationModeError(t('toast.imagenIncompatibleGenerationMode', { model: 'Imagen3' }));
  }

  const prompts = selectPresetModifiedPrompts(state);
  assert(prompts.positive.length > 0, 'Imagen3 requires positive prompt to have at least one character');

  const { originalSize, aspectRatio } = getOriginalAndScaledSizesForTextToImage(state);
  assert(isImagenAspectRatioID(aspectRatio.id), 'Imagen3 does not support this aspect ratio');

  const g = new Graph(getPrefixedId('imagen3_txt2img_graph'));
  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const imagen3 = g.addNode({
    // @ts-expect-error: These nodes are not available in the OSS application
    type: 'google_imagen3_generate_image',
    model: zModelIdentifierField.parse(model),
    negative_prompt: prompts.negative,
    aspect_ratio: aspectRatio.id,
    // When enhance_prompt is true, Imagen3 will return a new image every time, ignoring the seed.
    enhance_prompt: true,
    ...selectCanvasOutputFields(state),
  });

  g.addEdge(
    positivePrompt,
    'value',
    imagen3,
    // @ts-expect-error: These nodes are not available in the OSS application
    'positive_prompt'
  );
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  g.upsertMetadata({
    negative_prompt: prompts.negative,
    width: originalSize.width,
    height: originalSize.height,
    model: Graph.getModelMetadataField(model),
  });

  g.setMetadataReceivingNode(imagen3);

  return {
    g,
    positivePrompt,
  };
};
