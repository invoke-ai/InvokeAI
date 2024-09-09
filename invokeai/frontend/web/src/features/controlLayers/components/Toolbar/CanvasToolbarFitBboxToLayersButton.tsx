import { IconButton } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOut } from 'react-icons/pi';

export const CanvasToolbarFitBboxToLayersButton = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const onClick = useCallback(() => {
    canvasManager.bbox.fitToLayers();
  }, [canvasManager.bbox]);

  return (
    <IconButton
      onClick={onClick}
      variant="ghost"
      aria-label={t('controlLayers.fitBboxToLayers')}
      tooltip={t('controlLayers.fitBboxToLayers')}
      icon={<PiArrowsOut />}
    />
  );
});

CanvasToolbarFitBboxToLayersButton.displayName = 'CanvasToolbarFitBboxToLayersButton';
