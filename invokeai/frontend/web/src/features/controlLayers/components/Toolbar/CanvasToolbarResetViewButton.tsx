import { IconButton } from '@invoke-ai/ui-library';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';

export const CanvasToolbarResetViewButton = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const isCanvasFocused = useIsRegionFocused('canvas');

  const fitLayersToStage = useCallback(() => {
    canvasManager.stage.fitLayersToStage();
  }, [canvasManager.stage]);

  useRegisteredHotkeys({
    id: 'fitLayersToCanvas',
    category: 'canvas',
    callback: () => canvasManager.stage.fitLayersToStage(),
    options: { enabled: isCanvasFocused, preventDefault: true },
    dependencies: [isCanvasFocused],
  });
  useRegisteredHotkeys({
    id: 'fitBboxToCanvas',
    category: 'canvas',
    callback: () => canvasManager.stage.fitBboxToStage(),
    options: { enabled: isCanvasFocused, preventDefault: true },
    dependencies: [isCanvasFocused],
  });
  useRegisteredHotkeys({
    id: 'setZoomTo100Percent',
    category: 'canvas',
    callback: () => canvasManager.stage.setScale(1),
    options: { enabled: isCanvasFocused, preventDefault: true },
    dependencies: [isCanvasFocused],
  });
  useRegisteredHotkeys({
    id: 'setZoomTo200Percent',
    category: 'canvas',
    callback: () => canvasManager.stage.setScale(2),
    options: { enabled: isCanvasFocused, preventDefault: true },
    dependencies: [isCanvasFocused],
  });
  useRegisteredHotkeys({
    id: 'setZoomTo400Percent',
    category: 'canvas',
    callback: () => canvasManager.stage.setScale(4),
    options: { enabled: isCanvasFocused, preventDefault: true },
    dependencies: [isCanvasFocused],
  });
  useRegisteredHotkeys({
    id: 'setZoomTo800Percent',
    category: 'canvas',
    callback: () => canvasManager.stage.setScale(8),
    options: { enabled: isCanvasFocused, preventDefault: true },
    dependencies: [isCanvasFocused],
  });

  return (
    <IconButton
      tooltip={t('hotkeys.canvas.fitLayersToCanvas.title')}
      aria-label={t('hotkeys.canvas.fitLayersToCanvas.title')}
      onClick={fitLayersToStage}
      icon={<PiArrowsOutBold />}
      variant="link"
      alignSelf="stretch"
    />
  );
});

CanvasToolbarResetViewButton.displayName = 'CanvasToolbarResetViewButton';
