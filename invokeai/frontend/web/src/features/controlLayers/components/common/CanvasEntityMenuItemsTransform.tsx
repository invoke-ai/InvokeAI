import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityAdapter } from 'features/controlLayers/hooks/useEntityAdapter';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFrameCornersBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsTransform = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const adapter = useEntityAdapter(entityIdentifier);
  const isBusy = useCanvasIsBusy();

  const onClick = useCallback(() => {
    adapter.transformer.startTransform();
  }, [adapter.transformer]);

  return (
    <MenuItem onClick={onClick} icon={<PiFrameCornersBold />} isDisabled={isBusy}>
      {t('controlLayers.transform.transform')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsTransform.displayName = 'CanvasEntityMenuItemsTransform';
