import { useAppDispatch } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityDuplicated } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyFill } from 'react-icons/pi';

export const CanvasEntityMenuItemsDuplicate = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext();
  const isBusy = useCanvasIsBusy();

  const onClick = useCallback(() => {
    const editSession = canvasManager.tool.tools.path.$editSession.get();
    if (
      editSession?.entityIdentifier.id === entityIdentifier.id &&
      editSession.entityIdentifier.type === entityIdentifier.type
    ) {
      canvasManager.tool.tools.path.acceptEditSession();
    }
    dispatch(entityDuplicated({ entityIdentifier }));
  }, [canvasManager.tool.tools.path, dispatch, entityIdentifier]);

  return (
    <IconMenuItem
      aria-label={t('controlLayers.duplicate')}
      tooltip={t('controlLayers.duplicate')}
      onClick={onClick}
      icon={<PiCopyFill />}
      isDisabled={isBusy}
    />
  );
});

CanvasEntityMenuItemsDuplicate.displayName = 'CanvasEntityMenuItemsDuplicate';
