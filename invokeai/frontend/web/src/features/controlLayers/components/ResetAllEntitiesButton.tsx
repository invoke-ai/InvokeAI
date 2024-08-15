import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { allEntitiesDeleted } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const ResetAllEntitiesButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(allEntitiesDeleted());
  }, [dispatch]);

  return (
    <Button
      onClick={onClick}
      leftIcon={<PiTrashSimpleBold />}
      variant="ghost"
      colorScheme="error"
      data-testid="control-layers-delete-all-layers-button"
    >
      {t('controlLayers.resetAll')}
    </Button>
  );
});

ResetAllEntitiesButton.displayName = 'ResetAllEntitiesButton';
