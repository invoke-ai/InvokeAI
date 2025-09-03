import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityDeleted } from 'features/controlLayers/store/canvasInstanceSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type Props = {
  asIcon?: boolean;
};

export const CanvasEntityMenuItemsDelete = memo(({ asIcon = false }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const isBusy = useCanvasIsBusy();

  const deleteEntity = useCallback(() => {
    dispatch(entityDeleted({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

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
