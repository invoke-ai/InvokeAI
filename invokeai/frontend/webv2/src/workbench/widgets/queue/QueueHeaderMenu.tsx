import { Icon, Menu } from '@chakra-ui/react';
import { getApiErrorMessage } from '@workbench/backend/http';
import { useNotify } from '@workbench/useNotify';
import { Trash2Icon, TrashIcon } from 'lucide-react';
import { useCallback } from 'react';

import { requestQueueConfirmation } from './queueConfirmationStore';
import { refreshQueue, useQueueCounts } from './queueDataStore';
import { QueueMenuItems, useQueueMenuActions } from './queueMenuActions';
import { useQueueQueryScope } from './queueScope';
import { clearFailedQueueItems, clearScopedQueue } from './queueServerApi';

const ERROR_ITEM_HOVER_PROPS = { bg: 'bg.error', color: 'fg.error' };

/**
 * Queue actions contributed to the frame's shared actions menu via the manifest
 * `headerMenu` slot. This intentionally uses the same action model as the
 * topbar queue menu so backend mutations keep local run coordination in sync.
 */
export const QueueHeaderMenu = () => {
  const actions = useQueueMenuActions();
  const counts = useQueueCounts();
  const scope = useQueueQueryScope();
  const notify = useNotify();

  const clearFailed = useCallback(async () => {
    try {
      await clearFailedQueueItems(scope);
      await refreshQueue();
      notify.success('Failed queue items cleared');
    } catch (error) {
      notify.error('Failed to clear queue items', getApiErrorMessage(error, 'Could not clear failed queue items.'));
    }
  }, [notify, scope]);

  const clearAll = useCallback(async () => {
    try {
      await clearScopedQueue(scope);
      await refreshQueue();
      notify.success('Queue cleared');
    } catch (error) {
      notify.error('Failed to clear queue', getApiErrorMessage(error, 'Could not clear the queue.'));
    }
  }, [notify, scope]);
  const onClearFailed = useCallback(() => {
    requestQueueConfirmation({
      body: 'This permanently removes failed queue items in the current queue scope.',
      confirmLabel: 'Clear Failed Items',
      onConfirm: clearFailed,
      title: 'Clear failed queue items?',
    });
  }, [clearFailed]);
  const onClearAll = useCallback(() => {
    requestQueueConfirmation({
      body: 'This permanently clears queue items in the current queue scope.',
      confirmLabel: 'Clear Queue',
      onConfirm: clearAll,
      title: 'Clear queue?',
    });
  }, [clearAll]);

  return (
    <>
      <QueueMenuItems actions={actions} label="Queue" />
      <Menu.Separator />
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
          Clear
        </Menu.ItemGroupLabel>
        <Menu.Item
          color="fg.error"
          disabled={counts.failed === 0}
          value="clear-failed-items"
          _hover={ERROR_ITEM_HOVER_PROPS}
          onClick={onClearFailed}
        >
          <Icon as={Trash2Icon} boxSize="3" />
          <Menu.ItemText>Clear Failed Items</Menu.ItemText>
        </Menu.Item>
        <Menu.Item
          color="fg.error"
          disabled={counts.total === 0}
          value="clear-all-items"
          _hover={ERROR_ITEM_HOVER_PROPS}
          onClick={onClearAll}
        >
          <Icon as={TrashIcon} boxSize="3" />
          <Menu.ItemText>Clear All Items</Menu.ItemText>
        </Menu.Item>
      </Menu.ItemGroup>
    </>
  );
};
