import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaRedo } from 'react-icons/fa';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import {
  areHotkeysEnabledSelector,
  currentCanvasSelector,
  GenericCanvasState,
  InpaintingCanvasState,
  OutpaintingCanvasState,
  redo,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';

const canvasRedoSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (currentCanvas: GenericCanvasState, activeTabName, areHotkeysEnabled) => {
    const { futureObjects, shouldShowMask, tool } = currentCanvas as
      | InpaintingCanvasState
      | OutpaintingCanvasState;

    return {
      canRedo: futureObjects.length > 0,
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

export default function IAICanvasRedoControl() {
  const dispatch = useAppDispatch();
  const { canRedo, shouldShowMask, activeTabName, tool, areHotkeysEnabled } =
    useAppSelector(canvasRedoSelector);

  const handleRedo = () => {
    dispatch(redo());
  };

  // Hotkeys

  // Redo
  useHotkeys(
    'cmd+shift+z, control+shift+z, control+y, cmd+y',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleRedo();
    },
    {
      enabled: areHotkeysEnabled && canRedo,
    },
    [activeTabName, shouldShowMask, canRedo, tool]
  );

  return (
    <IAIIconButton
      aria-label="Redo"
      tooltip="Redo"
      icon={<FaRedo />}
      onClick={handleRedo}
      isDisabled={!canRedo || !shouldShowMask}
    />
  );
}
