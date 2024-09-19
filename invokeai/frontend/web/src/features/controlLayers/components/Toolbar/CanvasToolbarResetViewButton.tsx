import { $alt, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { INTERACTION_SCOPES } from 'common/hooks/interactionScopes';
import { $canvasManager } from 'features/controlLayers/store/ephemeral';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

export const CanvasToolbarResetViewButton = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useStore($canvasManager);
  const isCanvasActive = useStore(INTERACTION_SCOPES.canvas.$isActive);
  const imageViewer = useImageViewer();

  const resetZoom = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    canvasManager.stage.setScale(1);
  }, [canvasManager]);

  const resetView = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    canvasManager.stage.fitLayersToStage();
  }, [canvasManager]);

  const onReset = useCallback(() => {
    if ($alt.get()) {
      resetView();
    } else {
      resetZoom();
    }
  }, [resetView, resetZoom]);

  useRegisteredHotkeys({
    id: 'fitLayersToCanvas',
    category: 'canvas',
    callback: resetView,
    options: { enabled: isCanvasActive && !imageViewer.isOpen },
    dependencies: [isCanvasActive, imageViewer.isOpen],
  });
  useRegisteredHotkeys({
    id: 'setZoomTo100Percent',
    category: 'canvas',
    callback: resetZoom,
    options: { enabled: isCanvasActive && !imageViewer.isOpen },
    dependencies: [isCanvasActive, imageViewer.isOpen],
  });

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

CanvasToolbarResetViewButton.displayName = 'CanvasToolbarResetViewButton';
