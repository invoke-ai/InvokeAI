import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { $filteringEntity } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShootingStarBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsFilter = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const filter = useCallback(() => {
    $filteringEntity.set(entityIdentifier);
  }, [entityIdentifier]);

  return (
    <MenuItem onClick={filter} icon={<PiShootingStarBold />}>
      {t('controlLayers.filter.filter')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsFilter.displayName = 'CanvasEntityMenuItemsFilter';
