import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaUndo } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  areHotkeysEnabledSelector,
  currentCanvasSelector,
  GenericCanvasState,
  undo,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/optionsSelectors';

const canvasUndoSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (canvas: GenericCanvasState, activeTabName, areHotkeysEnabled) => {
    const { pastLines, shouldShowMask } = canvas;

    return {
      canUndo: pastLines.length > 0,
      shouldShowMask,
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

export default function IAICanvasUndoControl() {
  const dispatch = useAppDispatch();

  const { canUndo, shouldShowMask, activeTabName, areHotkeysEnabled } =
    useAppSelector(canvasUndoSelector);

  const handleUndo = () => dispatch(undo());

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
    [activeTabName, shouldShowMask, canUndo]
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
