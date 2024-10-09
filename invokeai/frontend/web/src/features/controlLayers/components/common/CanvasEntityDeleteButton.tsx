import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityDeleted } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi';

export const CanvasEntityDeleteButton = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const dispatch = useAppDispatch();
  const isBusy = useCanvasIsBusy();
  const onClick = useCallback(() => {
    dispatch(entityDeleted({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <IconButton
      size="sm"
      aria-label={t('common.delete')}
      tooltip={t('common.delete')}
      variant="link"
      alignSelf="stretch"
      icon={<PiTrashSimpleFill />}
      onClick={onClick}
      colorScheme="error"
      isDisabled={isBusy}
    />
  );
});

CanvasEntityDeleteButton.displayName = 'CanvasEntityDeleteButton';
