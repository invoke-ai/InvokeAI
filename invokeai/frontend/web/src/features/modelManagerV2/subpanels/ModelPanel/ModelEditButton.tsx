import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { setSelectedModelMode } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { IoPencil } from 'react-icons/io5';

export const ModelEditButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleEditModel = useCallback(() => {
    dispatch(setSelectedModelMode('edit'));
  }, [dispatch]);

  return (
    <Button size="sm" leftIcon={<IoPencil />} colorScheme="invokeYellow" onClick={handleEditModel} flexShrink={0}>
      {t('modelManager.edit')}
    </Button>
  );
};
