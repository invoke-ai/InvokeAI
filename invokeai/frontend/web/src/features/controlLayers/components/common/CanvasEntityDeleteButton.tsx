import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { stopPropagation } from 'common/util/stopPropagation';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { entityDeleted } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const CanvasEntityDeleteButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const onClick = useCallback(() => {
    dispatch(entityDeleted({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);
  return (
    <IconButton
      size="sm"
      colorScheme="error"
      aria-label={t('common.delete')}
      tooltip={t('common.delete')}
      icon={<PiTrashSimpleBold />}
      onClick={onClick}
      onDoubleClick={stopPropagation} // double click expands the layer
    />
  );
});

CanvasEntityDeleteButton.displayName = 'CanvasEntityDeleteButton';
