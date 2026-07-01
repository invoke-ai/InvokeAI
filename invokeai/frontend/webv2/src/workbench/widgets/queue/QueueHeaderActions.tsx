/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { Icon } from '@chakra-ui/react';
import { getApiErrorMessage } from '@workbench/backend/http';
import { Button } from '@workbench/components/ui';
import { useNotify } from '@workbench/useNotify';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { XIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueServerItem } from './queueServerApi';

import { refreshQueue, useNowNextItems } from './queueDataStore';
import { cancelQueueItem } from './queueServerApi';

export const getQueueHeaderCancelState = ({
  currentStatus,
  isConnected,
}: {
  currentStatus: QueueServerItem['status'] | null;
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
  const isConnected = useWorkbenchSelector((snapshot) => snapshot.state.backendConnection.status === 'connected');
  const { current } = useNowNextItems();
  const notify = useNotify();
  const [busy, setBusy] = useState(false);
  const currentItemId = current?.item_id ?? null;
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
      await cancelQueueItem(currentItemId);
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
