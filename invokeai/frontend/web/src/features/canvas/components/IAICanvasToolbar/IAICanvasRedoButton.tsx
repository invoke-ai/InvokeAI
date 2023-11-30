import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaRedo } from 'react-icons/fa';

import { redo } from 'features/canvas/store/canvasSlice';

import { stateSelector } from 'app/store/store';
import { isEqual } from 'lodash-es';
import { useTranslation } from 'react-i18next';
import { useCallback } from 'react';

const canvasRedoSelector = createSelector(
  [stateSelector, activeTabNameSelector],
  ({ canvas }, activeTabName) => {
    const { futureLayerStates } = canvas;

    return {
      canRedo: futureLayerStates.length > 0,
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

  const handleRedo = useCallback(() => {
    dispatch(redo());
  }, [dispatch]);

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
