import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityFilter } from 'features/controlLayers/hooks/useEntityFilter';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShootingStarBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsFilter = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const filter = useEntityFilter(entityIdentifier);

  return (
    <MenuItem onClick={filter.start} icon={<PiShootingStarBold />} isDisabled={filter.isDisabled}>
      {t('controlLayers.filter.filter')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsFilter.displayName = 'CanvasEntityMenuItemsFilter';
