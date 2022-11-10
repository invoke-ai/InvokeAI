import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaUndo } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { currentCanvasSelector, undo } from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/optionsSelectors';

const canvasUndoSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector],
  (canvas, activeTabName) => {
    const { pastObjects } = canvas;

    return {
      canUndo: pastObjects.length > 0,
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
    ['meta+z', 'control+z'],
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleUndo();
    },
    {
      enabled: canUndo,
    },
    [activeTabName, canUndo]
  );

  return (
    <IAIIconButton
      aria-label="Undo"
      tooltip="Undo"
      icon={<FaUndo />}
      onClick={handleUndo}
      isDisabled={!canUndo}
    />
  );
}
