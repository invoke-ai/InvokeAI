import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { isChatGPT4oAspectRatioID } from 'features/controlLayers/store/types';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  CANVAS_OUTPUT_PREFIX,
  getBoardField,
  selectPresetModifiedPrompts,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { t } from 'i18next';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

export const buildChatGPT4oGraph = async (state: RootState, manager: CanvasManager): Promise<GraphBuilderReturn> => {
  const generationMode = await manager.compositor.getGenerationMode();

  assert(
    generationMode === 'txt2img' || generationMode === 'img2img',
    t('toast.gptImageIncompatibleWithInpaintAndOutpaint')
  );

  log.debug({ generationMode }, 'Building GPT Image graph');

  const canvas = selectCanvasSlice(state);
  const canvasSettings = selectCanvasSettingsSlice(state);

  const { bbox } = canvas;
  const { positivePrompt } = selectPresetModifiedPrompts(state);

  assert(isChatGPT4oAspectRatioID(bbox.aspectRatio.id), 'ChatGPT 4o does not support this aspect ratio');

  const is_intermediate = canvasSettings.sendToCanvas;
  const board = canvasSettings.sendToCanvas ? undefined : getBoardField(state);

  if (generationMode === 'txt2img') {
    const g = new Graph(getPrefixedId('chatgpt_4o_txt2img_graph'));
    const gptImage = g.addNode({
      // @ts-expect-error: These nodes are not available in the OSS application
      type: 'chatgpt_4o_generate_image',
      id: getPrefixedId(CANVAS_OUTPUT_PREFIX),
      positive_prompt: positivePrompt,
      aspect_ratio: bbox.aspectRatio.id,
      use_cache: false,
      is_intermediate,
      board,
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
      positive_prompt: positivePrompt,
      image: { image_name },
      use_cache: false,
      is_intermediate,
      board,
    });
    return {
      g,
      positivePromptFieldIdentifier: { nodeId: gptImage.id, fieldName: 'positive_prompt' },
    };
  }

  assert<Equals<typeof generationMode, never>>(false, 'Invalid generation mode for gpt image');
};
