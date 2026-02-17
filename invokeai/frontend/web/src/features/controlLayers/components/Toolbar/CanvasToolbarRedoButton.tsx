import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { canvasRedo } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasMayRedo } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowClockwiseBold } from 'react-icons/pi';

export const CanvasToolbarRedoButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBusy = useCanvasIsBusy();
  const mayRedo = useAppSelector(selectCanvasMayRedo);
  const onClick = useCallback(() => {
    dispatch(canvasRedo());
  }, [dispatch]);

  return (
    <IAITooltip label={t('hotkeys.canvas.redo.title')}>
      <IconButton
        aria-label={t('hotkeys.canvas.redo.title')}
        onClick={onClick}
        icon={<PiArrowClockwiseBold />}
        variant="link"
        alignSelf="stretch"
        isDisabled={isBusy || !mayRedo}
      />
    </IAITooltip>
  );
});

CanvasToolbarRedoButton.displayName = 'CanvasToolbarRedoButton';
