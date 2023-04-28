import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaRedo } from 'react-icons/fa';

import { redo } from 'features/canvas/store/canvasSlice';
import { systemSelector } from 'features/system/store/systemSelectors';

import { isEqual } from 'lodash-es';
import { useTranslation } from 'react-i18next';

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
      resultEqualityCheck: isEqual,
    },
  }
);

export default function IAICanvasRedoButton() {
  const dispatch = useAppDispatch();
  const { canRedo, activeTabName } = useAppSelector(canvasRedoSelector);

  const { t } = useTranslation();

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
      aria-label={`${t('unifiedCanvas.redo')} (Ctrl+Shift+Z)`}
      tooltip={`${t('unifiedCanvas.redo')} (Ctrl+Shift+Z)`}
      icon={<FaRedo />}
      onClick={handleRedo}
      isDisabled={!canRedo}
    />
  );
}
