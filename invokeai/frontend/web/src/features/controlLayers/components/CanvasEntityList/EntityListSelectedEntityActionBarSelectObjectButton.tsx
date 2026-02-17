import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import { useEntitySegmentAnything } from 'features/controlLayers/hooks/useEntitySegmentAnything';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isSegmentableEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShapesFill } from 'react-icons/pi';

export const EntityListSelectedEntityActionBarSelectObjectButton = memo(() => {
  const { t } = useTranslation();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const segment = useEntitySegmentAnything(selectedEntityIdentifier);

  if (!selectedEntityIdentifier) {
    return null;
  }

  if (!isSegmentableEntityIdentifier(selectedEntityIdentifier)) {
    return null;
  }

  return (
    <IAITooltip label={t('controlLayers.selectObject.selectObject')}>
      <IconButton
        onClick={segment.start}
        isDisabled={segment.isDisabled}
        minW={8}
        variant="link"
        alignSelf="stretch"
        aria-label={t('controlLayers.selectObject.selectObject')}
        icon={<PiShapesFill />}
      />
    </IAITooltip>
  );
});

EntityListSelectedEntityActionBarSelectObjectButton.displayName = 'EntityListSelectedEntityActionBarSelectObjectButton';
