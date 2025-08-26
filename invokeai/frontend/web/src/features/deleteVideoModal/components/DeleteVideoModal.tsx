import { ConfirmationAlertDialog, Flex, FormControl, FormLabel, Switch, Text } from '@invoke-ai/ui-library';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useDeleteVideoModalApi, useDeleteVideoModalState } from 'features/deleteVideoModal/store/state';
import { selectSystemShouldConfirmOnDelete, setShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const DeleteVideoModal = memo(() => {
  const state = useDeleteVideoModalState();
  const api = useDeleteVideoModalApi();
  const { dispatch } = useAppStore();
  const { t } = useTranslation();
  const shouldConfirmOnDelete = useAppSelector(selectSystemShouldConfirmOnDelete);

  const handleChangeShouldConfirmOnDelete = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldConfirmOnDelete(!e.target.checked)),
    [dispatch]
  );

  return (
    <ConfirmationAlertDialog
      title={`${t('gallery.deleteVideo', { count: state.video_ids.length })}`}
      isOpen={state.isOpen}
      onClose={api.close}
      cancelButtonText={t('common.cancel')}
      acceptButtonText={t('common.delete')}
      acceptCallback={api.confirm}
      cancelCallback={api.cancel}
      useInert={false}
    >
      <Flex direction="column" gap={3}>
        <Text>{t('gallery.deleteVideoPermanent')}</Text>
        <Text>{t('common.areYouSure')}</Text>
        <FormControl>
          <FormLabel>{t('common.dontAskMeAgain')}</FormLabel>
          <Switch isChecked={!shouldConfirmOnDelete} onChange={handleChangeShouldConfirmOnDelete} />
        </FormControl>
      </Flex>
    </ConfirmationAlertDialog>
  );
});
DeleteVideoModal.displayName = 'DeleteVideoModal';
