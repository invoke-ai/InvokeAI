import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useSaveLayerToAssets } from 'features/controlLayers/hooks/useSaveLayerToAssets';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isSaveableEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

export const EntityListSelectedEntityActionBarSaveToAssetsButton = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const adapter = useEntityAdapterSafe(selectedEntityIdentifier);
  const saveLayerToAssets = useSaveLayerToAssets();
  const onClick = useCallback(() => {
    saveLayerToAssets(adapter);
  }, [saveLayerToAssets, adapter]);

  if (!selectedEntityIdentifier) {
    return null;
  }

  if (!isSaveableEntityIdentifier(selectedEntityIdentifier)) {
    return null;
  }

  return (
    <IconButton
      onClick={onClick}
      isDisabled={!selectedEntityIdentifier || isBusy}
      size="sm"
      variant="link"
      alignSelf="stretch"
      aria-label={t('controlLayers.saveLayerToAssets')}
      tooltip={t('controlLayers.saveLayerToAssets')}
      icon={<PiFloppyDiskBold />}
    />
  );
});

EntityListSelectedEntityActionBarSaveToAssetsButton.displayName = 'EntityListSelectedEntityActionBarSaveToAssetsButton';
