import { IconButton } from '@invoke-ai/ui-library';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useCanvasManager } from 'features/controlLayers/hooks/useCanvasManager';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiResizeBold } from 'react-icons/pi';

export const CanvasToolbarFitBboxToLayersButton = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const isBusy = useCanvasIsBusy();
  const isCanvasFocused = useIsRegionFocused('canvas');

  const onClick = useCallback(() => {
    canvasManager.tool.tools.bbox.fitToLayers();
    canvasManager.stage.fitLayersToStage();
  }, [canvasManager.tool.tools.bbox, canvasManager.stage]);

  useRegisteredHotkeys({
    id: 'fitBboxToLayers',
    category: 'canvas',
    callback: () => {
      canvasManager.tool.tools.bbox.fitToLayers();
      canvasManager.stage.fitLayersToStage();
    },
    options: { enabled: isCanvasFocused && !isBusy, preventDefault: true },
    dependencies: [isCanvasFocused, isBusy],
  });

  return (
    <IconButton
      onClick={onClick}
      variant="link"
      alignSelf="stretch"
      aria-label={t('controlLayers.fitBboxToLayers')}
      tooltip={t('controlLayers.fitBboxToLayers')}
      icon={<PiResizeBold />}
      isDisabled={isBusy}
    />
  );
});

CanvasToolbarFitBboxToLayersButton.displayName = 'CanvasToolbarFitBboxToLayersButton';
