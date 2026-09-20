import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityDeleted } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type Props = {
  asIcon?: boolean;
};

export const CanvasEntityMenuItemsDelete = memo(({ asIcon = false }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const entityIdentifier = useEntityIdentifierContext();
  const isBusy = useCanvasIsBusy();

  const deleteEntity = useCallback(() => {
    const editSession = canvasManager.tool.tools.path.$editSession.get();
    if (
      editSession?.entityIdentifier.id === entityIdentifier.id &&
      editSession.entityIdentifier.type === entityIdentifier.type
    ) {
      canvasManager.tool.tools.path.acceptEditSession();
    }
    dispatch(entityDeleted({ entityIdentifier }));
  }, [canvasManager.tool.tools.path, dispatch, entityIdentifier]);

  if (asIcon) {
    return (
      <IconMenuItem
        aria-label={t('common.delete')}
        tooltip={t('common.delete')}
        onClick={deleteEntity}
        icon={<PiTrashSimpleBold />}
        isDestructive
        isDisabled={isBusy}
      />
    );
  }

  return (
    <MenuItem onClick={deleteEntity} icon={<PiTrashSimpleBold />} isDestructive isDisabled={isBusy}>
      {t('common.delete')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsDelete.displayName = 'CanvasEntityMenuItemsDelete';
