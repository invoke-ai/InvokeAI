import { MenuItem } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { selectIsStaging } from 'features/controlLayers/store/canvasSessionSlice';
import { isFilterableEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShootingStarBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsFilter = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext();
  const isStaging = useAppSelector(selectIsStaging);
  const isBusy = useCanvasIsBusy();

  const onClick = useCallback(() => {
    if (!entityIdentifier) {
      return;
    }
    if (!isFilterableEntityIdentifier(entityIdentifier)) {
      return;
    }
    const adapter = canvasManager.getAdapter(entityIdentifier);
    if (!adapter) {
      return;
    }
    adapter.filterer.startFilter();
  }, [canvasManager, entityIdentifier]);

  return (
    <MenuItem onClick={onClick} icon={<PiShootingStarBold />} isDisabled={isBusy || isStaging}>
      {t('controlLayers.filter.filter')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsFilter.displayName = 'CanvasEntityMenuItemsFilter';
