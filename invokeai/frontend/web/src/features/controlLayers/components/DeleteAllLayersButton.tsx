import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { allLayersDeleted } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const DeleteAllLayersButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isDisabled = useAppSelector((s) => s.controlLayers.present.layers.length === 0);
  const onClick = useCallback(() => {
    dispatch(allLayersDeleted());
  }, [dispatch]);

  return (
    <Button
      onClick={onClick}
      leftIcon={<PiTrashSimpleBold />}
      variant="ghost"
      colorScheme="error"
      isDisabled={isDisabled}
    >
      {t('controlLayers.deleteAll')}
    </Button>
  );
});

DeleteAllLayersButton.displayName = 'DeleteAllLayersButton';
