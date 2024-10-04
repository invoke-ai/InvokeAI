import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useIsEntityInteractable } from 'features/controlLayers/hooks/useEntityIsInteractable';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCropBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsCropToBbox = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isInteractable = useIsEntityInteractable(entityIdentifier);
  const onClick = useCallback(() => {
    if (!adapter) {
      return;
    }
    adapter.cropToBbox();
  }, [adapter]);

  return (
    <MenuItem onClick={onClick} icon={<PiCropBold />} isDisabled={!isInteractable}>
      {t('controlLayers.cropLayerToBbox')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsCropToBbox.displayName = 'CanvasEntityMenuItemsCropToBbox';
