import { $shift, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

export const CanvasResetViewButton = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useStore($canvasManager);

  const resetZoom = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    canvasManager.setStageScale(1);
  }, [canvasManager]);

  const resetView = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    canvasManager.resetView();
  }, [canvasManager]);

  const onReset = useCallback(() => {
    if ($shift.get()) {
      resetZoom();
    } else {
      resetView();
    }
  }, [resetView, resetZoom]);

  useHotkeys('r', resetView);
  useHotkeys('shift+r', resetZoom);

  return (
    <IconButton
      tooltip={t('controlLayers.resetView')}
      aria-label={t('controlLayers.resetView')}
      onClick={onReset}
      icon={<PiArrowCounterClockwiseBold />}
      variant="link"
    />
  );
});

CanvasResetViewButton.displayName = 'CanvasResetViewButton';
