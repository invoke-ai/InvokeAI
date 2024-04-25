import type { RootState } from 'app/store/store';
import type { BoardField } from 'features/nodes/types/common';
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
 * Gets the SDXL style prompts, based on the concat setting.
 */
export const getSDXLStylePrompts = (state: RootState): { positiveStylePrompt: string; negativeStylePrompt: string } => {
  const { positivePrompt, negativePrompt } = state.regionalPrompts.present.baseLayer;
  const { positiveStylePrompt, negativeStylePrompt, shouldConcatSDXLStylePrompt } = state.sdxl;

  return {
    positiveStylePrompt: shouldConcatSDXLStylePrompt ? positivePrompt : positiveStylePrompt,
    negativeStylePrompt: shouldConcatSDXLStylePrompt ? negativePrompt : negativeStylePrompt,
  };
};

/**
 * Gets the is_intermediate field, based on the active tab and shouldAutoSave setting.
 */
export const getIsIntermediate = (state: RootState) => {
  const activeTabName = activeTabNameSelector(state);
  if (activeTabName === 'unifiedCanvas') {
    return !state.canvas.shouldAutoSave;
  }
  return false;
};
