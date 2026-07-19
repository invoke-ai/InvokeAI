import type { QueueItemReadModel } from '@features/queue/core/types';

/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { Icon } from '@chakra-ui/react';
import { queueCommands } from '@features/queue/publicApi';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { XIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { refreshQueue, useNowNextItems } from './queueDataStore';
import { useQueueUi } from './QueueUiContext';

export const getQueueHeaderCancelState = ({
  currentStatus,
  isConnected,
}: {
  currentStatus: QueueItemReadModel['status'] | null;
  isConnected: boolean;
}): { disabled: boolean; itemLabel: string } => ({
  disabled: !isConnected || currentStatus !== 'in_progress',
  itemLabel: 'widgets.queue.cancelCurrent',
});

/**
 * Cancel the current queue item from the manifest `headerActions` slot, so the
 * most common interruption action sits inline beside the actions menu.
 */
export const QueueHeaderActions = () => {
  const { t } = useTranslation();
  const { isConnected, notify } = useQueueUi();
  const { current } = useNowNextItems();
  const [busy, setBusy] = useState(false);
  const currentItemId = current?.id ?? null;
  const { disabled, itemLabel } = getQueueHeaderCancelState({
    currentStatus: current?.status ?? null,
    isConnected,
  });

  const onCancel = async () => {
    if (disabled || currentItemId === null) {
      return;
    }

    setBusy(true);

    try {
      await queueCommands.cancelItem(currentItemId);
      await refreshQueue();
    } catch (error) {
      notify.error(t('common.cancelFailed'), getApiErrorMessage(error, t('widgets.queue.couldNotCancelCurrentItem')));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Button disabled={disabled} loading={busy} size="2xs" variant="outline" onClick={onCancel}>
      <Icon as={XIcon} boxSize="3.5" />
      {t(itemLabel)}
    </Button>
  );
};
