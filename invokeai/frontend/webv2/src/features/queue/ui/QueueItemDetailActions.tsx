import type { QueueItemReadModel } from '@features/queue/core/types';

import { Icon } from '@chakra-ui/react';
import { queueCommands } from '@features/queue/publicApi';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { LayersIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { requestQueueConfirmation } from './queueConfirmationStore';
import { refreshQueue } from './queueDataStore';
import { useQueueUi } from './QueueUiContext';

/**
 * Queue-owned actions of an expanded row. Cancel batch stops every unfinished sibling of this item without touching
 * other batches; it stays available on finished rows because their siblings may still be queued.
 */
export const QueueItemDetailActions = ({ item }: { item: QueueItemReadModel }) => {
  const { t } = useTranslation();
  const { canManageItem, notify } = useQueueUi();
  const [busy, setBusy] = useState(false);

  const cancelBatch = useCallback(async () => {
    const owner = captureAccountScope();
    setBusy(true);

    try {
      const canceled = await queueCommands.cancelBatch(item.batchId);
      if (!isAccountScopeCurrent(owner)) {
        return;
      }
      await refreshQueue();
      if (canceled === 0) {
        notify.info(t('widgets.queue.batchNothingToCancel'), t('widgets.queue.batchNothingToCancelDescription'));
      } else {
        notify.success(
          t('widgets.queue.batchCancellationRequested'),
          t('widgets.queue.batchCanceledCount', { count: canceled })
        );
      }
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(
          t('widgets.queue.failedToCancelBatch'),
          getApiErrorMessage(error, t('widgets.queue.couldNotCancelBatch'))
        );
      }
    } finally {
      setBusy(false);
    }
  }, [item.batchId, notify, t]);

  const onCancelBatch = useCallback(() => {
    requestQueueConfirmation({
      body: t('widgets.queue.cancelBatchConfirmationBody'),
      confirmLabel: t('widgets.queue.cancelBatch'),
      onConfirm: cancelBatch,
      title: t('widgets.queue.cancelBatchTitle'),
    });
  }, [cancelBatch, t]);

  if (!canManageItem(item)) {
    return null;
  }

  return (
    <Button loading={busy} size="sm" variant="subtle" onClick={onCancelBatch}>
      <Icon as={LayersIcon} boxSize="3" />
      {t('widgets.queue.cancelBatch')}
    </Button>
  );
};
