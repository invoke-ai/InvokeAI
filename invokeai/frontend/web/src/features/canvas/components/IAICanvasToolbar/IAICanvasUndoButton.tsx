import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { undo } from 'features/canvas/store/canvasSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaUndo } from 'react-icons/fa';

const canvasUndoSelector = createMemoizedSelector(
  [stateSelector, activeTabNameSelector],
  ({ canvas }, activeTabName) => {
    const { pastLayerStates } = canvas;

    return {
      canUndo: pastLayerStates.length > 0,
      activeTabName,
    };
  }
);

export default function IAICanvasUndoButton() {
  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const { canUndo, activeTabName } = useAppSelector(canvasUndoSelector);

  const handleUndo = useCallback(() => {
    dispatch(undo());
  }, [dispatch]);

  useHotkeys(
    ['meta+z', 'ctrl+z'],
    () => {
      handleUndo();
    },
    {
      enabled: () => canUndo,
      preventDefault: true,
    },
    [activeTabName, canUndo]
  );

  return (
    <IAIIconButton
      aria-label={`${t('unifiedCanvas.undo')} (Ctrl+Z)`}
      tooltip={`${t('unifiedCanvas.undo')} (Ctrl+Z)`}
      icon={<FaUndo />}
      onClick={handleUndo}
      isDisabled={!canUndo}
    />
  );
}
