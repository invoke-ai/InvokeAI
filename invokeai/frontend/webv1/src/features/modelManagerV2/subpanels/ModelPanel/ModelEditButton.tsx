import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { setSelectedModelMode } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPencilBold } from 'react-icons/pi';

export const ModelEditButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleEditModel = useCallback(() => {
    dispatch(setSelectedModelMode('edit'));
  }, [dispatch]);

  return (
    <Button size="sm" leftIcon={<PiPencilBold />} colorScheme="invokeYellow" onClick={handleEditModel} flexShrink={0}>
      {t('modelManager.edit')}
    </Button>
  );
});

ModelEditButton.displayName = 'ModelEditButton';
