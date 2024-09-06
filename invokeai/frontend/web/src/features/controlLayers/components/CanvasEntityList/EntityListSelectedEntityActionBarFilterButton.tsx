import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isFilterableEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShootingStarBold } from 'react-icons/pi';

export const EntityListSelectedEntityActionBarFilterButton = memo(() => {
  const { t } = useTranslation();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const canvasManager = useCanvasManager();
  const isBusy = useCanvasIsBusy();

  const onClick = useCallback(() => {
    if (!selectedEntityIdentifier) {
      return;
    }
    if (!isFilterableEntityIdentifier(selectedEntityIdentifier)) {
      return;
    }

    canvasManager.filter.startFilter(selectedEntityIdentifier);
  }, [canvasManager, selectedEntityIdentifier]);

  if (!selectedEntityIdentifier) {
    return null;
  }

  if (!isFilterableEntityIdentifier(selectedEntityIdentifier)) {
    return null;
  }

  return (
    <IconButton
      onClick={onClick}
      isDisabled={isBusy}
      size="sm"
      variant="link"
      alignSelf="stretch"
      aria-label={t('controlLayers.filter.filter')}
      tooltip={t('controlLayers.filter.filter')}
      icon={<PiShootingStarBold />}
    />
  );
});

EntityListSelectedEntityActionBarFilterButton.displayName = 'EntityListSelectedEntityActionBarFilterButton';
