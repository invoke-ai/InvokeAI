import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntitySegmentAnything } from 'features/controlLayers/hooks/useEntitySegmentAnything';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isSegmentableEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMaskHappyBold } from 'react-icons/pi';

export const EntityListSelectedEntityActionBarAutoMaskButton = memo(() => {
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
    <IconButton
      onClick={segment.start}
      isDisabled={segment.isDisabled}
      size="sm"
      variant="link"
      alignSelf="stretch"
      aria-label={t('controlLayers.segment.autoMask')}
      tooltip={t('controlLayers.segment.autoMask')}
      icon={<PiMaskHappyBold />}
    />
  );
});

EntityListSelectedEntityActionBarAutoMaskButton.displayName = 'EntityListSelectedEntityActionBarAutoMaskButton';
