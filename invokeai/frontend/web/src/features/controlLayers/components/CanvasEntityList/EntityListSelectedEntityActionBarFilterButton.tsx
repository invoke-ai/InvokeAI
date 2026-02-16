import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import { useEntityFilter } from 'features/controlLayers/hooks/useEntityFilter';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isFilterableEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShootingStarFill } from 'react-icons/pi';

export const EntityListSelectedEntityActionBarFilterButton = memo(() => {
  const { t } = useTranslation();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const filter = useEntityFilter(selectedEntityIdentifier);

  if (!selectedEntityIdentifier) {
    return null;
  }

  if (!isFilterableEntityIdentifier(selectedEntityIdentifier)) {
    return null;
  }

  return (
    <IAITooltip label={t('controlLayers.filter.filter')}>
      <IconButton
        onClick={filter.start}
        isDisabled={filter.isDisabled}
        minW={8}
        variant="link"
        alignSelf="stretch"
        aria-label={t('controlLayers.filter.filter')}
        icon={<PiShootingStarFill />}
      />
    </IAITooltip>
  );
});

EntityListSelectedEntityActionBarFilterButton.displayName = 'EntityListSelectedEntityActionBarFilterButton';
