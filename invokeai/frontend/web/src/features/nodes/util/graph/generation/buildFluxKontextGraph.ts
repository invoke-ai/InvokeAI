import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { isFluxKontextReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import type { ImageField } from 'features/nodes/types/common';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  CANVAS_OUTPUT_PREFIX,
  getBoardField,
  selectPresetModifiedPrompts,
} from 'features/nodes/util/graph/graphBuilderUtils';
import { type GraphBuilderReturn, UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { t } from 'i18next';
import { selectMainModelConfig } from 'services/api/endpoints/models';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

export const buildFluxKontextGraph = async (state: RootState, manager: CanvasManager): Promise<GraphBuilderReturn> => {
  const generationMode = await manager.compositor.getGenerationMode();

  if (generationMode !== 'txt2img') {
    throw new UnsupportedGenerationModeError(t('toast.fluxKontextIncompatibleGenerationMode'));
  }

  log.debug({ generationMode }, 'Building Flux Kontext graph');

  const model = selectMainModelConfig(state);

  const canvas = selectCanvasSlice(state);
  const canvasSettings = selectCanvasSettingsSlice(state);

  const { bbox } = canvas;
  const { positivePrompt } = selectPresetModifiedPrompts(state);

  assert(model, 'No model found in state');
  assert(model.base === 'flux-kontext', 'Model is not a Flux Kontext model');

  const is_intermediate = canvasSettings.sendToCanvas;
  const board = canvasSettings.sendToCanvas ? undefined : getBoardField(state);

  const validRefImages = canvas.referenceImages.entities
    .filter((entity) => entity.isEnabled)
    .filter((entity) => isFluxKontextReferenceImageConfig(entity.ipAdapter))
    .filter((entity) => getGlobalReferenceImageWarnings(entity, model).length === 0);

  let input_image: ImageField | undefined = undefined;

  if (validRefImages[0]) {
    assert(validRefImages.length === 1, 'Flux Kontext can have at most one reference image');

    assert(validRefImages[0].ipAdapter.image, 'Image is required for reference image');
    input_image = {
      image_name: validRefImages[0].ipAdapter.image.image_name,
    };
  }

  if (generationMode === 'txt2img') {
    const g = new Graph(getPrefixedId('flux_kontext_txt2img_graph'));
    const fluxKontextImage = g.addNode({
      // @ts-expect-error: These nodes are not available in the OSS application
      type: input_image ? 'flux_kontext_edit_image' : 'flux_kontext_generate_image',
      id: getPrefixedId(CANVAS_OUTPUT_PREFIX),
      model: zModelIdentifierField.parse(model),
      positive_prompt: positivePrompt,
      aspect_ratio: bbox.aspectRatio.id,
      use_cache: false,
      is_intermediate,
      board,
      input_image,
      prompt_upsampling: true,
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
  }

  assert<Equals<typeof generationMode, never>>(false, 'Invalid generation mode for Flux Kontext');
};
