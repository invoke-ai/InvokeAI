import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityAdapter } from 'features/controlLayers/hooks/useEntityAdapter';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFrameCornersBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsTransform = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const canvasManager = useCanvasManager();
  const adapter = useEntityAdapter(entityIdentifier);
  const transformingEntity = useStore(canvasManager.stateApi.$transformingEntity);

  const onClick = useCallback(() => {
    adapter.transformer.startTransform();
  }, [adapter.transformer]);

  return (
    <MenuItem onClick={onClick} icon={<PiFrameCornersBold />} isDisabled={Boolean(transformingEntity)}>
      {t('controlLayers.transform.transform')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsTransform.displayName = 'CanvasEntityMenuItemsTransform';
