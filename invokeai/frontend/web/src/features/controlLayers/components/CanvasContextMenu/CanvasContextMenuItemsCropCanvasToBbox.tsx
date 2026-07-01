import { MenuItem } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCropBold } from 'react-icons/pi';

export const CanvasContextMenuItemsCropCanvasToBbox = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const canvasManager = useCanvasManager();
  const cropCanvasToBbox = useCallback(async () => {
    const adapters = canvasManager.getAllAdapters();
    for (const adapter of adapters) {
      await adapter.cropToBbox();
    }
  }, [canvasManager]);

  return (
    <MenuItem icon={<PiCropBold />} isDisabled={isBusy} onClick={cropCanvasToBbox}>
      {t('controlLayers.canvasContextMenu.cropCanvasToBbox')}
    </MenuItem>
  );
});

CanvasContextMenuItemsCropCanvasToBbox.displayName = 'CanvasContextMenuItemsCropCanvasToBbox';
