import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaRedo } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import {
  areHotkeysEnabledSelector,
  currentCanvasSelector,
  redo,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';

const canvasRedoSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (currentCanvas, activeTabName, areHotkeysEnabled) => {
    const { futureObjects } = currentCanvas;

    return {
      canRedo: futureObjects.length > 0,
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

export default function IAICanvasRedoButton() {
  const dispatch = useAppDispatch();
  const { canRedo, activeTabName, areHotkeysEnabled } =
    useAppSelector(canvasRedoSelector);

  const handleRedo = () => {
    dispatch(redo());
  };

  useHotkeys(
    ['meta+shift+z', 'control+shift+z', 'control+y', 'meta+y'],
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleRedo();
    },
    {
      enabled: areHotkeysEnabled && canRedo,
    },
    [activeTabName, canRedo]
  );

  return (
    <IAIIconButton
      aria-label="Redo"
      tooltip="Redo"
      icon={<FaRedo />}
      onClick={handleRedo}
      isDisabled={!canRedo}
    />
  );
}
