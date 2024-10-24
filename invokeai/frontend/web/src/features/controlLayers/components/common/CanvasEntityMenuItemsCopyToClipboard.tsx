import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCopyLayerToClipboard } from 'features/controlLayers/hooks/useCopyLayerToClipboard';
import { useEntityIsEmpty } from 'features/controlLayers/hooks/useEntityIsEmpty';
import { useIsEntityInteractable } from 'features/controlLayers/hooks/useEntityIsInteractable';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsCopyToClipboard = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isInteractable = useIsEntityInteractable(entityIdentifier);
  const isEmpty = useEntityIsEmpty(entityIdentifier);
  const copyLayerToClipboard = useCopyLayerToClipboard();

  const onClick = useCallback(() => {
    copyLayerToClipboard(adapter);
  }, [copyLayerToClipboard, adapter]);

  return (
    <MenuItem onClick={onClick} icon={<PiCopyBold />} isDisabled={!isInteractable || isEmpty}>
      {t('common.clipboard')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsCopyToClipboard.displayName = 'CanvasEntityMenuItemsCopyToClipboard';
