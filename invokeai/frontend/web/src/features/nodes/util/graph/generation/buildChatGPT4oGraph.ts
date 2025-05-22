import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { isChatGPT4oAspectRatioID, isChatGPT4oReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { type ImageField, zModelIdentifierField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  CANVAS_OUTPUT_PREFIX,
  getBoardField,
  selectPresetModifiedPrompts,
} from 'features/nodes/util/graph/graphBuilderUtils';
import { type GraphBuilderReturn, UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { t } from 'i18next';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

export const buildChatGPT4oGraph = async (state: RootState, manager: CanvasManager): Promise<GraphBuilderReturn> => {
  const generationMode = await manager.compositor.getGenerationMode();

  if (generationMode !== 'txt2img' && generationMode !== 'img2img') {
    throw new UnsupportedGenerationModeError(t('toast.chatGPT4oIncompatibleGenerationMode'));
  }

  log.debug({ generationMode }, 'Building GPT Image graph');

  const model = selectMainModelConfig(state);

  const canvas = selectCanvasSlice(state);
  const canvasSettings = selectCanvasSettingsSlice(state);

  const { bbox } = canvas;
  const { positivePrompt } = selectPresetModifiedPrompts(state);

  assert(model, 'No model found in state');
  assert(model.base === 'chatgpt-4o', 'Model is not a ChatGPT 4o model');

  assert(isChatGPT4oAspectRatioID(bbox.aspectRatio.id), 'ChatGPT 4o does not support this aspect ratio');

  const validRefImages = canvas.referenceImages.entities
    .filter((entity) => entity.isEnabled)
    .filter((entity) => isChatGPT4oReferenceImageConfig(entity.ipAdapter))
    .filter((entity) => getGlobalReferenceImageWarnings(entity, model).length === 0)
    .toReversed(); // sends them in order they are displayed in the list

  let reference_images: ImageField[] | undefined = undefined;

  if (validRefImages.length > 0) {
    reference_images = [];
    for (const entity of validRefImages) {
      assert(entity.ipAdapter.image, 'Image is required for reference image');
      reference_images.push({
        image_name: entity.ipAdapter.image.image_name,
      });
    }
  }

  const is_intermediate = canvasSettings.sendToCanvas;
  const board = canvasSettings.sendToCanvas ? undefined : getBoardField(state);

  if (generationMode === 'txt2img') {
    const g = new Graph(getPrefixedId('chatgpt_4o_txt2img_graph'));
    const gptImage = g.addNode({
      // @ts-expect-error: These nodes are not available in the OSS application
      type: 'chatgpt_4o_generate_image',
      id: getPrefixedId(CANVAS_OUTPUT_PREFIX),
      model: zModelIdentifierField.parse(model),
      positive_prompt: positivePrompt,
      aspect_ratio: bbox.aspectRatio.id,
      reference_images,
      use_cache: false,
      is_intermediate,
      board,
    });
    g.upsertMetadata({
      positive_prompt: positivePrompt,
      model: Graph.getModelMetadataField(model),
      width: bbox.rect.width,
      height: bbox.rect.height,
    });
    return {
      g,
      positivePromptFieldIdentifier: { nodeId: gptImage.id, fieldName: 'positive_prompt' },
    };
  }

  if (generationMode === 'img2img') {
    const adapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
    const { image_name } = await manager.compositor.getCompositeImageDTO(adapters, bbox.rect, {
      is_intermediate: true,
      silent: true,
    });
    const g = new Graph(getPrefixedId('chatgpt_4o_img2img_graph'));
    const gptImage = g.addNode({
      // @ts-expect-error: These nodes are not available in the OSS application
      type: 'chatgpt_4o_edit_image',
      id: getPrefixedId(CANVAS_OUTPUT_PREFIX),
      model: zModelIdentifierField.parse(model),
      positive_prompt: positivePrompt,
      aspect_ratio: bbox.aspectRatio.id,
      base_image: { image_name },
      reference_images,
      use_cache: false,
      is_intermediate,
      board,
    });
    g.upsertMetadata({
      positive_prompt: positivePrompt,
      model: Graph.getModelMetadataField(model),
      width: bbox.rect.width,
      height: bbox.rect.height,
    });
    return {
      g,
      positivePromptFieldIdentifier: { nodeId: gptImage.id, fieldName: 'positive_prompt' },
    };
  }

  assert<Equals<typeof generationMode, never>>(false, 'Invalid generation mode for ChatGPT ');
};
