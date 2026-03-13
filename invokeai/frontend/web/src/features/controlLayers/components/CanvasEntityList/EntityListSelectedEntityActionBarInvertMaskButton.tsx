import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useInvertMask } from 'features/controlLayers/hooks/useInvertMask';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isMaskEntityIdentifier } from 'features/controlLayers/store/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSelectionInverseBold } from 'react-icons/pi';

export const EntityListSelectedEntityActionBarInvertMaskButton = memo(() => {
  const { t } = useTranslation();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const invertMask = useInvertMask();

  if (!selectedEntityIdentifier) {
    return null;
  }

  if (!isMaskEntityIdentifier(selectedEntityIdentifier)) {
    return null;
  }

  const label =
    selectedEntityIdentifier.type === 'regional_guidance'
      ? t('controlLayers.invertRegion', { defaultValue: 'Invert Region' })
      : t('controlLayers.invertMask');

  return (
    <IAITooltip label={label}>
      <IconButton
        onClick={invertMask}
        isDisabled={isBusy}
        minW={8}
        variant="link"
        alignSelf="stretch"
        aria-label={label}
        icon={<PiSelectionInverseBold />}
      />
    </IAITooltip>
  );
});

EntityListSelectedEntityActionBarInvertMaskButton.displayName = 'EntityListSelectedEntityActionBarInvertMaskButton';
