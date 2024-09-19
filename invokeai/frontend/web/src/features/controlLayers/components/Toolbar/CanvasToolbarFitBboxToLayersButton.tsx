import { IconButton } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiResizeBold } from 'react-icons/pi';

export const CanvasToolbarFitBboxToLayersButton = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const isBusy = useCanvasIsBusy();
  const onClick = useCallback(() => {
    canvasManager.bbox.fitToLayers();
  }, [canvasManager.bbox]);

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
