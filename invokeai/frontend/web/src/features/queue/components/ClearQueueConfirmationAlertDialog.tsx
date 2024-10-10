import { ConfirmationAlertDialog, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { listCursorChanged, listPriorityChanged } from 'features/queue/store/queueSlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useClearQueueMutation, useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

const [useClearQueueConfirmationAlertDialog] = buildUseBoolean(false);

export const useClearQueue = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const dialog = useClearQueueConfirmationAlertDialog();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const isConnected = useStore($isConnected);
  const [trigger, { isLoading }] = useClearQueueMutation({
    fixedCacheKey: 'clearQueue',
  });

  const clearQueue = useCallback(async () => {
    if (!queueStatus?.queue.total) {
      return;
    }

    try {
      await trigger().unwrap();
      toast({
        id: 'QUEUE_CLEAR_SUCCEEDED',
        title: t('queue.clearSucceeded'),
        status: 'success',
      });
      dispatch(listCursorChanged(undefined));
      dispatch(listPriorityChanged(undefined));
    } catch {
      toast({
        id: 'QUEUE_CLEAR_FAILED',
        title: t('queue.clearFailed'),
        status: 'error',
      });
    }
  }, [queueStatus?.queue.total, trigger, dispatch, t]);

  const isDisabled = useMemo(() => !isConnected || !queueStatus?.queue.total, [isConnected, queueStatus?.queue.total]);

  return {
    clearQueue,
    isOpen: dialog.isTrue,
    openDialog: dialog.setTrue,
    closeDialog: dialog.setFalse,
    isLoading,
    queueStatus,
    isDisabled,
  };
};

export const ClearQueueConfirmationsAlertDialog = memo(() => {
  useAssertSingleton('ClearQueueConfirmationsAlertDialog');
  const { t } = useTranslation();
  const clearQueue = useClearQueue();

  return (
    <ConfirmationAlertDialog
      isOpen={clearQueue.isOpen}
      onClose={clearQueue.closeDialog}
      title={t('queue.clearTooltip')}
      acceptCallback={clearQueue.clearQueue}
      acceptButtonText={t('queue.clear')}
      useInert={false}
    >
      <Text>{t('queue.clearQueueAlertDialog')}</Text>
      <br />
      <Text>{t('queue.clearQueueAlertDialog2')}</Text>
    </ConfirmationAlertDialog>
  );
});

ClearQueueConfirmationsAlertDialog.displayName = 'ClearQueueConfirmationsAlertDialog';
