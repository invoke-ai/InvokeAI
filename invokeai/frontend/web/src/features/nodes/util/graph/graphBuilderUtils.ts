import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { type ParamsState, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import type { CanvasState } from 'features/controlLayers/store/types';
import type { BoardField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { buildPresetModifiedPrompt } from 'features/stylePresets/hooks/usePresetModifiedPrompts';
import { selectStylePresetSlice } from 'features/stylePresets/store/stylePresetSlice';
import { pick } from 'lodash-es';
import { selectListStylePresetsRequestState } from 'services/api/endpoints/stylePresets';
import type { Invocation } from 'services/api/types';
import { assert } from 'tsafe';

import type { MainModelLoaderNodes } from './types';

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
 * Gets the prompts, modified for the active style preset.
 */
export const selectPresetModifiedPrompts = createSelector(
  selectParamsSlice,
  selectStylePresetSlice,
  selectListStylePresetsRequestState,
  (params, stylePresetSlice, listStylePresetsRequestState) => {
    const { positivePrompt, negativePrompt, positivePrompt2, negativePrompt2, shouldConcatPrompts } = params;
    const { activeStylePresetId } = stylePresetSlice;

    if (activeStylePresetId) {
      const { data } = listStylePresetsRequestState;

      const activeStylePreset = data?.find((item) => item.id === activeStylePresetId);

      if (activeStylePreset) {
        const presetModifiedPositivePrompt = buildPresetModifiedPrompt(
          activeStylePreset.preset_data.positive_prompt,
          positivePrompt
        );

        const presetModifiedNegativePrompt = buildPresetModifiedPrompt(
          activeStylePreset.preset_data.negative_prompt,
          negativePrompt
        );

        return {
          positivePrompt: presetModifiedPositivePrompt,
          negativePrompt: presetModifiedNegativePrompt,
          positiveStylePrompt: shouldConcatPrompts ? presetModifiedPositivePrompt : positivePrompt2,
          negativeStylePrompt: shouldConcatPrompts ? presetModifiedNegativePrompt : negativePrompt2,
        };
      }
    }

    return {
      positivePrompt,
      negativePrompt,
      positiveStylePrompt: shouldConcatPrompts ? positivePrompt : positivePrompt2,
      negativeStylePrompt: shouldConcatPrompts ? negativePrompt : negativePrompt2,
    };
  }
);

export const getSizes = (bboxState: CanvasState['bbox']) => {
  const originalSize = pick(bboxState.rect, 'width', 'height');
  const scaledSize = ['auto', 'manual'].includes(bboxState.scaleMethod) ? bboxState.scaledSize : originalSize;
  return { originalSize, scaledSize };
};

export const getInfill = (
  g: Graph,
  params: ParamsState
): Invocation<'infill_patchmatch' | 'infill_cv2' | 'infill_lama' | 'infill_rgba' | 'infill_tile'> => {
  const { infillMethod, infillColorValue, infillPatchmatchDownscaleSize, infillTileSize } = params;

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
    const { a, ...rgb } = infillColorValue;
    const color = { ...rgb, a: Math.round(a * 255) };
    return g.addNode({
      id: 'infill_rgba',
      type: 'infill_rgba',
      color,
    });
  }

  assert(false, 'Unknown infill method');
};

export const CANVAS_OUTPUT_PREFIX = 'canvas_output';

export const isMainModelWithoutUnet = (modelLoader: Invocation<MainModelLoaderNodes>) => {
  return (
    modelLoader.type === 'flux_model_loader' ||
    modelLoader.type === 'sd3_model_loader' ||
    modelLoader.type === 'cogview4_model_loader'
  );
};
