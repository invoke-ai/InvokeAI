import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaUndo } from 'react-icons/fa';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  areHotkeysEnabledSelector,
  currentCanvasSelector,
  GenericCanvasState,
  OutpaintingCanvasState,
  undo,
  undoEraser,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/optionsSelectors';

const canvasUndoSelector = createSelector(
  [
    (state: RootState) => state.canvas.outpainting,
    currentCanvasSelector,
    activeTabNameSelector,
    areHotkeysEnabledSelector,
  ],
  (
    outpainting: OutpaintingCanvasState,
    canvas: GenericCanvasState,
    activeTabName,
    areHotkeysEnabled
  ) => {
    const { pastLines, shouldShowMask, tool } = canvas;

    return {
      canUndo:
        tool === 'imageEraser'
          ? outpainting.pastEraserLines.length > 0
          : pastLines.length > 0,
      shouldShowMask,
      activeTabName,
      areHotkeysEnabled,
      tool,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function IAICanvasUndoControl() {
  const dispatch = useAppDispatch();

  const { canUndo, shouldShowMask, activeTabName, areHotkeysEnabled, tool } =
    useAppSelector(canvasUndoSelector);

  const handleUndo = () => {
    if (tool === 'imageEraser') {
      dispatch(undoEraser());
    } else {
      dispatch(undo());
    }
  };

  // Hotkeys
  // Undo
  useHotkeys(
    'cmd+z, control+z',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleUndo();
    },
    {
      enabled: areHotkeysEnabled && canUndo,
    },
    [activeTabName, shouldShowMask, canUndo, tool]
  );

  return (
    <IAIIconButton
      aria-label="Undo"
      tooltip="Undo"
      icon={<FaUndo />}
      onClick={handleUndo}
      isDisabled={!canUndo || !shouldShowMask}
    />
  );
}
