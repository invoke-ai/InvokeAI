import { IconButton } from '@invoke-ai/ui-library';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useSaveLayerToAssets } from 'features/controlLayers/hooks/useSaveLayerToAssets';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

export const CanvasEntitySaveToAssets = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const saveLayerToAssets = useSaveLayerToAssets();
  const onClick = useCallback(() => {
    saveLayerToAssets(adapter);
  }, [saveLayerToAssets, adapter]);

  return (
    <IconButton
      size="sm"
      aria-label={t('controlLayers.saveLayerToAssets')}
      tooltip={t('controlLayers.saveLayerToAssets')}
      variant="link"
      alignSelf="stretch"
      icon={<PiFloppyDiskBold />}
      onClick={onClick}
      isDisabled={isBusy}
    />
  );
});

CanvasEntitySaveToAssets.displayName = 'CanvasEntitySaveToAssets';
