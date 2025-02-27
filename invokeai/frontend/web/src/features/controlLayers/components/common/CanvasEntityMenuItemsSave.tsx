import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useSaveLayerToAssets } from 'features/controlLayers/hooks/useSaveLayerToAssets';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsSave = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const saveLayerToAssets = useSaveLayerToAssets();
  const onClick = useCallback(() => {
    saveLayerToAssets(adapter);
  }, [saveLayerToAssets, adapter]);

  return (
    <MenuItem onClick={onClick} icon={<PiFloppyDiskBold />} isDisabled={isBusy}>
      {t('controlLayers.saveLayerToAssets')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsSave.displayName = 'CanvasEntityMenuItemsSave';
