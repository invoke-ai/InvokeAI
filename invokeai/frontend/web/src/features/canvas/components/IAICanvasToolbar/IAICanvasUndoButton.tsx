import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { undo } from 'features/canvas/store/canvasSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

const IAICanvasUndoButton = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const canUndo = useAppSelector((s) => s.canvas.pastLayerStates.length > 0);

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
    <IconButton
      aria-label={`${t('unifiedCanvas.undo')} (Ctrl+Z)`}
      tooltip={`${t('unifiedCanvas.undo')} (Ctrl+Z)`}
      icon={<PiArrowCounterClockwiseBold />}
      onClick={handleUndo}
      isDisabled={!canUndo}
    />
  );
};

export default memo(IAICanvasUndoButton);
