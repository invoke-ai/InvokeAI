import type { RootState } from 'app/store/store';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import type { BoardField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { pick } from 'lodash-es';
import type { Invocation } from 'services/api/types';
import { assert } from 'tsafe';

/**
 * Gets the board field, based on the autoAddBoardId setting.
 */
export const getBoardField = (state: RootState): BoardField | undefined => {
  const { autoAddBoardId } = state.gallery;
  if (autoAddBoardId === 'none') {
    return undefined;
  }
  return { board_id: autoAddBoardId };
};

/**
 * Gets the SDXL style prompts, based on the concat setting.
 */
export const getSDXLStylePrompts = (state: RootState): { positiveStylePrompt: string; negativeStylePrompt: string } => {
  const { positivePrompt, negativePrompt, positivePrompt2, negativePrompt2, shouldConcatPrompts } = state.canvasV2;

  return {
    positiveStylePrompt: shouldConcatPrompts ? positivePrompt : positivePrompt2,
    negativeStylePrompt: shouldConcatPrompts ? negativePrompt : negativePrompt2,
  };
};

/**
 * Gets the is_intermediate field, based on the active tab and shouldAutoSave setting.
 */
export const getIsIntermediate = (state: RootState) => {
  const activeTabName = activeTabNameSelector(state);
  if (activeTabName === 'canvas') {
    return !state.canvas.shouldAutoSave;
  }
  return false;
};

export const getSizes = (bboxState: CanvasV2State['bbox']) => {
  const originalSize = pick(bboxState, 'width', 'height');
  const scaledSize = ['auto', 'manual'].includes(bboxState.scaleMethod) ? bboxState.scaledSize : originalSize;
  return { originalSize, scaledSize };
};

export const getInfill = (
  g: Graph,
  compositing: CanvasV2State['compositing']
): Invocation<'infill_patchmatch' | 'infill_cv2' | 'infill_lama' | 'infill_rgba' | 'infill_tile'> => {
  const { infillMethod, infillColorValue, infillPatchmatchDownscaleSize, infillTileSize } = compositing;

  // Add Infill Nodes
  if (infillMethod === 'patchmatch') {
    return g.addNode({
      id: 'infill_patchmatch',
      type: 'infill_patchmatch',
      downscale: infillPatchmatchDownscaleSize,
    });
  }

  if (infillMethod === 'lama') {
    return g.addNode({
      id: 'infill_lama',
      type: 'infill_lama',
    });
  }

  if (infillMethod === 'cv2') {
    return g.addNode({
      id: 'infill_cv2',
      type: 'infill_cv2',
    });
  }

  if (infillMethod === 'tile') {
    return g.addNode({
      id: 'infill_tile',
      type: 'infill_tile',
      tile_size: infillTileSize,
    });
  }

  if (infillMethod === 'color') {
    return g.addNode({
      id: 'infill_rgba',
      type: 'infill_rgba',
      color: infillColorValue,
    });
  }

  assert(false, 'Unknown infill method');
};
