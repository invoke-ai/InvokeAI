import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { canvasUndo } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasMayUndo } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

export const CanvasToolbarUndoButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBusy = useCanvasIsBusy();
  const mayUndo = useAppSelector(selectCanvasMayUndo);
  const onClick = useCallback(() => {
    dispatch(canvasUndo());
  }, [dispatch]);

  return (
    <IAITooltip label={t('hotkeys.canvas.undo.title')}>
      <IconButton
        aria-label={t('hotkeys.canvas.undo.title')}
        onClick={onClick}
        icon={<PiArrowCounterClockwiseBold />}
        variant="link"
        alignSelf="stretch"
        isDisabled={isBusy || !mayUndo}
      />
    </IAITooltip>
  );
});

CanvasToolbarUndoButton.displayName = 'CanvasToolbarUndoButton';
