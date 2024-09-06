import { MenuItem } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { isControlLayerEntityIdentifier, isRasterLayerEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShootingStarBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsFilter = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext();
  const isBusy = useCanvasIsBusy();

  const onClick = useCallback(() => {
    if (!entityIdentifier) {
      return;
    }
    // Can only filter raster and control layers
    if (!isRasterLayerEntityIdentifier(entityIdentifier) && !isControlLayerEntityIdentifier(entityIdentifier)) {
      return;
    }
    canvasManager.filter.startFilter(entityIdentifier);
  }, [canvasManager.filter, entityIdentifier]);

  return (
    <MenuItem onClick={onClick} icon={<PiShootingStarBold />} isDisabled={isBusy}>
      {t('controlLayers.filter.filter')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsFilter.displayName = 'CanvasEntityMenuItemsFilter';
