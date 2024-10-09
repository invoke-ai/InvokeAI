import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityTransform } from 'features/controlLayers/hooks/useEntityTransform';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isTransformableEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFrameCornersBold } from 'react-icons/pi';

export const EntityListSelectedEntityActionBarTransformButton = memo(() => {
  const { t } = useTranslation();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const transform = useEntityTransform(selectedEntityIdentifier);

  if (!selectedEntityIdentifier) {
    return null;
  }

  if (!isTransformableEntityIdentifier(selectedEntityIdentifier)) {
    return null;
  }

  return (
    <IconButton
      onClick={transform.start}
      isDisabled={transform.isDisabled}
      size="sm"
      variant="link"
      alignSelf="stretch"
      aria-label={t('controlLayers.transform.transform')}
      tooltip={t('controlLayers.transform.transform')}
      icon={<PiFrameCornersBold />}
    />
  );
});

EntityListSelectedEntityActionBarTransformButton.displayName = 'EntityListSelectedEntityActionBarTransformButton';
