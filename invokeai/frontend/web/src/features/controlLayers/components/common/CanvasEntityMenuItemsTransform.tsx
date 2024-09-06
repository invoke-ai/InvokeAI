import { MenuItem } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityAdapter } from 'features/controlLayers/hooks/useEntityAdapter';
import { selectIsStaging } from 'features/controlLayers/store/canvasSessionSlice';
import { isTransformableEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFrameCornersBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsTransform = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const adapter = useEntityAdapter(entityIdentifier);
  const isStaging = useAppSelector(selectIsStaging);
  const isBusy = useCanvasIsBusy();

  const onClick = useCallback(() => {
    if (!isTransformableEntityIdentifier(entityIdentifier)) {
      return;
    }
    adapter.transformer.startTransform();
  }, [adapter.transformer, entityIdentifier]);

  return (
    <MenuItem onClick={onClick} icon={<PiFrameCornersBold />} isDisabled={isBusy || isStaging}>
      {t('controlLayers.transform.transform')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsTransform.displayName = 'CanvasEntityMenuItemsTransform';
