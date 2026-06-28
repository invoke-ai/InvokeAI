import { Icon } from '@chakra-ui/react';
import { getApiErrorMessage } from '@workbench/backend/http';
import { Button } from '@workbench/components/ui';
import { useNotify } from '@workbench/useNotify';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { XIcon } from 'lucide-react';
import { useCallback, useState } from 'react';

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
  itemLabel: 'Cancel Current',
});

/**
 * Cancel the current queue item from the manifest `headerActions` slot, so the
 * most common interruption action sits inline beside the actions menu.
 */
export const QueueHeaderActions = () => {
  const isConnected = useWorkbenchSelector((snapshot) => snapshot.state.backendConnection.status === 'connected');
  const { current } = useNowNextItems();
  const notify = useNotify();
  const [busy, setBusy] = useState(false);
  const { disabled, itemLabel } = getQueueHeaderCancelState({
    currentStatus: current?.status ?? null,
    isConnected,
  });

  const onCancel = useCallback(async () => {
    if (disabled || !current) {
      return;
    }

    setBusy(true);

    try {
      await cancelQueueItem(current.item_id);
      await refreshQueue();
    } catch (error) {
      notify.error('Cancel failed', getApiErrorMessage(error, 'Could not cancel the current item.'));
    } finally {
      setBusy(false);
    }
  }, [current, disabled, notify]);

  return (
    <Button disabled={disabled} loading={busy} size="2xs" variant="outline" onClick={onCancel}>
      <Icon as={XIcon} boxSize="3.5" />
      {itemLabel}
    </Button>
  );
};
