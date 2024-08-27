import { IconButton, useShiftModifier } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { allEntitiesDeleted, entityDeleted } from 'features/controlLayers/store/canvasSlice';
import { selectEntityCount, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi';

export const EntityListActionBarDeleteButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const entityCount = useAppSelector(selectEntityCount);
  const shift = useShiftModifier();
  const onClick = useCallback(() => {
    if (shift) {
      dispatch(allEntitiesDeleted());
      return;
    }
    if (!selectedEntityIdentifier) {
      return;
    }
    dispatch(entityDeleted({ entityIdentifier: selectedEntityIdentifier }));
  }, [dispatch, selectedEntityIdentifier, shift]);

  return (
    <IconButton
      onClick={onClick}
      isDisabled={shift ? entityCount === 0 : !selectedEntityIdentifier}
      size="sm"
      variant="ghost"
      aria-label={shift ? t('controlLayers.deleteAll') : t('controlLayers.deleteSelected')}
      tooltip={shift ? t('controlLayers.deleteAll') : t('controlLayers.deleteSelected')}
      icon={<PiTrashSimpleFill />}
    />
  );
});

EntityListActionBarDeleteButton.displayName = 'EntityListActionBarDeleteButton';
