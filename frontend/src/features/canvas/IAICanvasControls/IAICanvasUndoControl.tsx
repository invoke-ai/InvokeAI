import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaUndo } from 'react-icons/fa';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  areHotkeysEnabledSelector,
  currentCanvasSelector,
  GenericCanvasState,
  InpaintingCanvasState,
  OutpaintingCanvasState,
  undo,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/optionsSelectors';

const canvasUndoSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (canvas: GenericCanvasState, activeTabName, areHotkeysEnabled) => {
    const { pastObjects, shouldShowMask, tool } = canvas as
      | InpaintingCanvasState
      | OutpaintingCanvasState;

    return {
      canUndo: pastObjects.length > 0,
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
    dispatch(undo());
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
