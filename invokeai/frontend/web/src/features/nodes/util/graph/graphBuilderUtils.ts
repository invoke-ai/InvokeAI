import type { RootState } from 'app/store/store';
import type { BoardField } from 'features/nodes/types/common';
import { buildPresetModifiedPrompt } from 'features/stylePresets/hooks/usePresetModifiedPrompts';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';

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
export const getPresetModifiedPrompts = (state: RootState): { positivePrompt: string; negativePrompt: string, positiveStylePrompt?: string; negativeStylePrompt?: string } => {
  const { positivePrompt, negativePrompt, positivePrompt2, negativePrompt2, shouldConcatPrompts } =
    state.controlLayers.present;
  const { activeStylePreset } = state.stylePreset

  if (activeStylePreset) {
    const presetModifiedPositivePrompt = buildPresetModifiedPrompt(activeStylePreset.preset_data.positive_prompt, positivePrompt)

    const presetModifiedNegativePrompt = buildPresetModifiedPrompt(activeStylePreset.preset_data.negative_prompt, negativePrompt)

    return {
      positivePrompt: presetModifiedPositivePrompt,
      negativePrompt: presetModifiedNegativePrompt,
      positiveStylePrompt: shouldConcatPrompts ? presetModifiedPositivePrompt : positivePrompt2,
      negativeStylePrompt: shouldConcatPrompts ? presetModifiedNegativePrompt : negativePrompt2,

    };
  }

  return {
    positivePrompt,
    negativePrompt,
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
