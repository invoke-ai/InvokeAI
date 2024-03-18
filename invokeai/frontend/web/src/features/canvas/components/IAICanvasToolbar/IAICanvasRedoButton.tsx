import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { redo } from 'features/canvas/store/canvasSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowClockwiseBold } from 'react-icons/pi';

const IAICanvasRedoButton = () => {
  const dispatch = useAppDispatch();
  const canRedo = useAppSelector((s) => s.canvas.futureLayerStates.length > 0);
  const activeTabName = useAppSelector(activeTabNameSelector);

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
    <IconButton
      aria-label={`${t('unifiedCanvas.redo')} (Ctrl+Shift+Z)`}
      tooltip={`${t('unifiedCanvas.redo')} (Ctrl+Shift+Z)`}
      icon={<PiArrowClockwiseBold />}
      onClick={handleRedo}
      isDisabled={!canRedo}
    />
  );
};

export default memo(IAICanvasRedoButton);
