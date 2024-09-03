import { MenuItem } from '@invoke-ai/ui-library';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShootingStarBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsFilter = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext();
  const isBusy = useCanvasIsBusy();

  const onClick = useCallback(() => {
    canvasManager.filter.initialize(entityIdentifier);
  }, [canvasManager.filter, entityIdentifier]);

  return (
    <MenuItem onClick={onClick} icon={<PiShootingStarBold />} isDisabled={isBusy}>
      {t('controlLayers.filter.filter')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsFilter.displayName = 'CanvasEntityMenuItemsFilter';
