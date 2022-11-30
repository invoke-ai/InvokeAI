import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaUndo } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import { undo } from 'features/canvas/store/canvasSlice';
import { systemSelector } from 'features/system/store/systemSelectors';

const canvasUndoSelector = createSelector(
  [canvasSelector, activeTabNameSelector, systemSelector],
  (canvas, activeTabName, system) => {
    const { pastLayerStates } = canvas;

    return {
      canUndo: pastLayerStates.length > 0 && !system.isProcessing,
      activeTabName,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function IAICanvasUndoButton() {
  const dispatch = useAppDispatch();

  const { canUndo, activeTabName } = useAppSelector(canvasUndoSelector);

  const handleUndo = () => {
    dispatch(undo());
  };

  useHotkeys(
    ['meta+z', 'ctrl+z'],
    () => {
      handleUndo();
    },
    {
      enabled: () => canUndo,
      preventDefault: true,
    },
    [activeTabName, canUndo]
  );

  return (
    <IAIIconButton
      aria-label="Undo (Ctrl+Z)"
      tooltip="Undo (Ctrl+Z)"
      icon={<FaUndo />}
      onClick={handleUndo}
      isDisabled={!canUndo}
    />
  );
}
