import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectSaveAllImagesToGallery } from 'features/controlLayers/store/canvasSettingsSlice';
import {
  selectImg2imgStrength,
  selectMainModelConfig,
  selectOptimizedDenoisingEnabled,
  selectParamsSlice,
  selectRefinerModel,
  selectRefinerStart,
} from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { ParamsState } from 'features/controlLayers/store/types';
import type { BoardField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { buildPresetModifiedPrompt } from 'features/stylePresets/hooks/usePresetModifiedPrompts';
import { selectStylePresetSlice } from 'features/stylePresets/store/stylePresetSlice';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
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
 * Builds the common fields for canvas output:
 * - id
 * - use_cache
 * - is_intermediate
 * - board
 */
export const selectCanvasOutputFields = (state: RootState) => {
  // Advanced session means working on canvas - images are not saved to gallery or added to a board.
  // Simple session means working in YOLO mode - images are saved to gallery & board.
  const tab = selectActiveTab(state);
  const saveAllImagesToGallery = selectSaveAllImagesToGallery(state);

  // If we're on canvas and the save all images setting is enabled, save to gallery
  const is_intermediate = tab === 'canvas' && !saveAllImagesToGallery;
  const board = tab === 'canvas' && !saveAllImagesToGallery ? undefined : getBoardField(state);

  return {
    is_intermediate,
    board,
    use_cache: false,
    id: getPrefixedId(CANVAS_OUTPUT_PREFIX),
  };
};

/**
 * Gets the prompts, modified for the active style preset.
 */
export const selectPresetModifiedPrompts = createSelector(
  selectParamsSlice,
  selectStylePresetSlice,
  selectListStylePresetsRequestState,
  (params, stylePresetSlice, listStylePresetsRequestState) => {
    const negativePrompt = params.negativePrompt ?? '';
    const { positivePrompt, positivePrompt2, negativePrompt2, shouldConcatPrompts } = params;
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
          negativePrompt ?? ''
        );

        return {
          positive: presetModifiedPositivePrompt,
          negative: presetModifiedNegativePrompt,
          positiveStyle: positivePrompt2,
          negativeStyle: negativePrompt2,
          useMainPromptsForStyle: shouldConcatPrompts,
        };
      }
    }

    return {
      positive: positivePrompt,
      negative: negativePrompt,
      positiveStyle: positivePrompt2,
      negativeStyle: negativePrompt2,
      useMainPromptsForStyle: shouldConcatPrompts,
    };
  }
);

export const getOriginalAndScaledSizesForTextToImage = (state: RootState) => {
  const tab = selectActiveTab(state);
  const params = selectParamsSlice(state);
  const canvas = selectCanvasSlice(state);

  if (tab === 'canvas') {
    const { rect, aspectRatio } = canvas.bbox;
    const { width, height } = rect;
    const originalSize = { width, height };
    const scaledSize = ['auto', 'manual'].includes(canvas.bbox.scaleMethod) ? canvas.bbox.scaledSize : originalSize;
    return { originalSize, scaledSize, aspectRatio };
  } else if (tab === 'generate') {
    const { rect, aspectRatio } = params.dimensions;
    const { width, height } = rect;
    return {
      originalSize: { width, height },
      scaledSize: { width, height },
      aspectRatio,
    };
  }

  assert(false, `Cannot get sizes for tab ${tab} - this function is only for the Canvas or Generate tabs`);
};

export const getOriginalAndScaledSizesForOtherModes = (state: RootState) => {
  const tab = selectActiveTab(state);
  const canvas = selectCanvasSlice(state);

  assert(tab === 'canvas', `Cannot get sizes for tab ${tab} - this function is only for the Canvas tab`);

  const { rect, aspectRatio } = canvas.bbox;
  const { width, height } = rect;
  const originalSize = { width, height };
  const scaledSize = ['auto', 'manual'].includes(canvas.bbox.scaleMethod) ? canvas.bbox.scaledSize : originalSize;

  return { originalSize, scaledSize, aspectRatio, rect };
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

const CANVAS_OUTPUT_PREFIX = 'canvas_output';

export const isMainModelWithoutUnet = (modelLoader: Invocation<MainModelLoaderNodes>) => {
  return (
    modelLoader.type === 'flux_model_loader' ||
    modelLoader.type === 'sd3_model_loader' ||
    modelLoader.type === 'cogview4_model_loader'
  );
};

export const isCanvasOutputNodeId = (nodeId: string) => nodeId.split(':')[0] === CANVAS_OUTPUT_PREFIX;

export const getDenoisingStartAndEnd = (state: RootState): { denoising_start: number; denoising_end: number } => {
  const optimizedDenoisingEnabled = selectOptimizedDenoisingEnabled(state);
  const denoisingStrength = selectImg2imgStrength(state);
  const model = selectMainModelConfig(state);
  const refinerModel = selectRefinerModel(state);
  const refinerDenoisingStart = selectRefinerStart(state);

  switch (model?.base) {
    case 'sd-3': {
      // We rescale the img2imgStrength (with exponent 0.2) to effectively use the entire range [0, 1] and make the scale
      // more user-friendly for SD3.5. Without this, most of the 'change' is concentrated in the high denoise strength
      // range (>0.9).
      const exponent = optimizedDenoisingEnabled ? 0.2 : 1;
      return {
        denoising_start: 1 - denoisingStrength ** exponent,
        denoising_end: 1,
      };
    }
    case 'flux': {
      if (model.variant === 'inpaint') {
        // This is a FLUX Fill model - we always denoise fully
        return {
          denoising_start: 0,
          denoising_end: 1,
        };
      } else {
        // We rescale the img2imgStrength (with exponent 0.2) to effectively use the entire range [0, 1] and make the scale
        // more user-friendly for SD3.5. Without this, most of the 'change' is concentrated in the high denoise strength
        // range (>0.9).
        const exponent = optimizedDenoisingEnabled ? 0.2 : 1;
        return {
          denoising_start: 1 - denoisingStrength ** exponent,
          denoising_end: 1,
        };
      }
    }
    case 'sd-1':
    case 'sd-2':
    case 'cogview4': {
      return {
        denoising_start: 1 - denoisingStrength,
        denoising_end: 1,
      };
    }
    case 'sdxl': {
      if (refinerModel) {
        return {
          denoising_start: Math.min(refinerDenoisingStart, 1 - denoisingStrength),
          denoising_end: refinerDenoisingStart,
        };
      } else {
        return {
          denoising_start: 1 - denoisingStrength,
          denoising_end: 1,
        };
      }
    }
    default: {
      assert(false, `Unsupported base: ${model?.base}`);
    }
  }
};
