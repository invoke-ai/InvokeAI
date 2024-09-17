import { IconButton } from '@invoke-ai/ui-library';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useCopyLayerToClipboard } from 'features/controlLayers/hooks/useCopyLayerToClipboard';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

export const CanvasEntityCopyToClipboard = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const copyLayerToClipboard = useCopyLayerToClipboard();
  const onClick = useCallback(() => {
    copyLayerToClipboard(adapter);
  }, [copyLayerToClipboard, adapter]);

  return (
    <IconButton
      size="sm"
      aria-label={t('unifiedCanvas.copyToClipboard')}
      tooltip={t('unifiedCanvas.copyToClipboard')}
      variant="link"
      alignSelf="stretch"
      icon={<PiCopyBold />}
      onClick={onClick}
      isDisabled={isBusy}
    />
  );
});

CanvasEntityCopyToClipboard.displayName = 'CanvasEntityCopyToClipboard';
