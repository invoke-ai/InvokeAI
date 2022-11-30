import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaRedo } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';

import _ from 'lodash';
import { redo } from 'features/canvas/store/canvasSlice';
import { systemSelector } from 'features/system/store/systemSelectors';

const canvasRedoSelector = createSelector(
  [canvasSelector, activeTabNameSelector, systemSelector],
  (canvas, activeTabName, system) => {
    const { futureLayerStates } = canvas;

    return {
      canRedo: futureLayerStates.length > 0 && !system.isProcessing,
      activeTabName,
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
  const { canRedo, activeTabName } = useAppSelector(canvasRedoSelector);

  const handleRedo = () => {
    dispatch(redo());
  };

  useHotkeys(
    ['meta+shift+z', 'ctrl+shift+z', 'control+y', 'meta+y'],
    () => {
      handleRedo();
    },
    {
      enabled: () => canRedo,
      preventDefault: true,
    },
    [activeTabName, canRedo]
  );

  return (
    <IAIIconButton
      aria-label="Redo (Ctrl+Shift+Z)"
      tooltip="Redo (Ctrl+Shift+Z)"
      icon={<FaRedo />}
      onClick={handleRedo}
      isDisabled={!canRedo}
    />
  );
}
