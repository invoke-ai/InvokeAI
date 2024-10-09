import { useAppDispatch } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useIsEntityInteractable } from 'features/controlLayers/hooks/useEntityIsInteractable';
import { entityDeleted } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsDelete = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const isInteractable = useIsEntityInteractable(entityIdentifier);

  const deleteEntity = useCallback(() => {
    dispatch(entityDeleted({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <IconMenuItem
      aria-label={t('common.delete')}
      tooltip={t('common.delete')}
      onClick={deleteEntity}
      icon={<PiTrashSimpleBold />}
      isDestructive
      isDisabled={!isInteractable}
    />
  );
});

CanvasEntityMenuItemsDelete.displayName = 'CanvasEntityMenuItemsDelete';
