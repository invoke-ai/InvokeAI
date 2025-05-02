import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { isImagen3AspectRatioID } from 'features/controlLayers/store/types';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  CANVAS_OUTPUT_PREFIX,
  getBoardField,
  selectPresetModifiedPrompts,
} from 'features/nodes/util/graph/graphBuilderUtils';
import { type GraphBuilderReturn, UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { t } from 'i18next';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

export const buildImagen3Graph = async (state: RootState, manager: CanvasManager): Promise<GraphBuilderReturn> => {
  const generationMode = await manager.compositor.getGenerationMode();

  if (generationMode !== 'txt2img') {
    throw new UnsupportedGenerationModeError(t('toast.imagen3IncompatibleGenerationMode'));
  }

  log.debug({ generationMode }, 'Building Imagen3 graph');

  const canvas = selectCanvasSlice(state);
  const canvasSettings = selectCanvasSettingsSlice(state);

  const { bbox } = canvas;
  const { positivePrompt, negativePrompt } = selectPresetModifiedPrompts(state);

  assert(isImagen3AspectRatioID(bbox.aspectRatio.id), 'Imagen3 does not support this aspect ratio');
  assert(positivePrompt.length > 0, 'Imagen3 requires positive prompt to have at least one character');

  const is_intermediate = canvasSettings.sendToCanvas;
  const board = canvasSettings.sendToCanvas ? undefined : getBoardField(state);

  if (generationMode === 'txt2img') {
    const g = new Graph(getPrefixedId('imagen3_txt2img_graph'));
    const imagen3 = g.addNode({
      // @ts-expect-error: These nodes are not available in the OSS application
      type: 'google_imagen3_generate_image',
      id: getPrefixedId(CANVAS_OUTPUT_PREFIX),
      positive_prompt: positivePrompt,
      negative_prompt: negativePrompt,
      aspect_ratio: bbox.aspectRatio.id,
      enhance_prompt: true,
      // When enhance_prompt is true, Imagen3 will return a new image every time, ignoring the seed.
      use_cache: false,
      is_intermediate,
      board,
    });
    return {
      g,
      seedFieldIdentifier: { nodeId: imagen3.id, fieldName: 'seed' },
      positivePromptFieldIdentifier: { nodeId: imagen3.id, fieldName: 'positive_prompt' },
    };
  }

  assert<Equals<typeof generationMode, never>>(false, 'Invalid generation mode for Imagen3');
};
