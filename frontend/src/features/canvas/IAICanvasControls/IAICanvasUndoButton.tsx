import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaUndo } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  areHotkeysEnabledSelector,
  currentCanvasSelector,
  undo,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/optionsSelectors';

const canvasUndoSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (canvas, activeTabName, areHotkeysEnabled) => {
    const { pastObjects } = canvas;

    return {
      canUndo: pastObjects.length > 0,
      activeTabName,
      areHotkeysEnabled,
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

  const { canUndo, activeTabName, areHotkeysEnabled } =
    useAppSelector(canvasUndoSelector);

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
      enabled: areHotkeysEnabled && canUndo,
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
