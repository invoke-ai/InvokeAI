import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { redo } from 'features/canvas/store/canvasSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaRedo } from 'react-icons/fa';

const canvasRedoSelector = createMemoizedSelector(
  [stateSelector, activeTabNameSelector],
  ({ canvas }, activeTabName) => {
    const { futureLayerStates } = canvas;

    return {
      canRedo: futureLayerStates.length > 0,
      activeTabName,
    };
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
