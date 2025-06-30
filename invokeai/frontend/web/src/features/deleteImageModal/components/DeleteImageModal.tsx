import { ConfirmationAlertDialog, Divider, Flex, FormControl, FormLabel, Switch, Text } from '@invoke-ai/ui-library';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import ImageUsageMessage from 'features/deleteImageModal/components/ImageUsageMessage';
import { useDeleteImageModalApi, useDeleteImageModalState } from 'features/deleteImageModal/store/state';
import { selectSystemShouldConfirmOnDelete, setShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const DeleteImageModal = memo(() => {
  const state = useDeleteImageModalState();
  const api = useDeleteImageModalApi();
  const { dispatch } = useAppStore();
  const { t } = useTranslation();
  const shouldConfirmOnDelete = useAppSelector(selectSystemShouldConfirmOnDelete);

  const handleChangeShouldConfirmOnDelete = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldConfirmOnDelete(!e.target.checked)),
    [dispatch]
  );

  return (
    <ConfirmationAlertDialog
      title={`${t('gallery.deleteImage', { count: state.image_names.length })}2`}
      isOpen={state.isOpen}
      onClose={api.close}
      cancelButtonText={t('common.cancel')}
      acceptButtonText={t('common.delete')}
      acceptCallback={api.confirm}
      cancelCallback={api.cancel}
      useInert={false}
    >
      <Flex direction="column" gap={3}>
        <ImageUsageMessage imageUsage={state.usageSummary} />
        <Divider />
        <Text>{t('gallery.deleteImagePermanent')}</Text>
        <Text>{t('common.areYouSure')}</Text>
        <FormControl>
          <FormLabel>{t('common.dontAskMeAgain')}</FormLabel>
          <Switch isChecked={!shouldConfirmOnDelete} onChange={handleChangeShouldConfirmOnDelete} />
        </FormControl>
      </Flex>
    </ConfirmationAlertDialog>
  );
});
DeleteImageModal.displayName = 'DeleteImageModal';
