import { IconButton } from '@invoke-ai/ui-library';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useExportCanvasToPSD } from 'features/controlLayers/hooks/useExportCanvasToPSD';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFileArrowDownBold } from 'react-icons/pi';

export const RasterLayerExportPSDButton = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const { exportCanvasToPSD } = useExportCanvasToPSD();

  const onClick = useCallback(() => {
    exportCanvasToPSD();
  }, [exportCanvasToPSD]);

  return (
    <IconButton
      onClick={onClick}
      isDisabled={isBusy}
      size="sm"
      variant="link"
      colorScheme="invokeBlue"
      alignSelf="stretch"
      aria-label={t('controlLayers.exportCanvasToPSD')}
      tooltip={t('controlLayers.exportCanvasToPSD')}
      icon={<PiFileArrowDownBold />}
    />
  );
});

RasterLayerExportPSDButton.displayName = 'RasterLayerExportPSDButton';
