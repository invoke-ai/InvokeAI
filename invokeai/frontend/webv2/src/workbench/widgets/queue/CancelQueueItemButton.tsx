import type { MouseEvent as ReactMouseEvent } from 'react';

import { Icon } from '@chakra-ui/react';
import { getApiErrorMessage } from '@workbench/backend/http';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { useNotify } from '@workbench/useNotify';
import { XIcon } from 'lucide-react';
import { useCallback, useState } from 'react';

import { refreshQueue } from './queueDataStore';
import { cancelQueueItem } from './queueServerApi';

/**
 * Cancel a single queue item by its backend id. Works for any in-flight item
 * regardless of which client submitted it; the coordinator settles local items
 * off the resulting `queue_item_status_changed` event. `stopPropagation` keeps a
 * click from also toggling the row it sits in.
 */
export const CancelQueueItemButton = ({ itemId }: { itemId: number }) => {
  const notify = useNotify();
  const [busy, setBusy] = useState(false);

  const onCancel = useCallback(
    async (event: ReactMouseEvent) => {
      event.stopPropagation();
      setBusy(true);

      try {
        await cancelQueueItem(itemId);
        await refreshQueue();
      } catch (error) {
        notify.error('Cancel failed', getApiErrorMessage(error, 'Could not cancel this item.'));
      } finally {
        setBusy(false);
      }
    },
    [itemId, notify]
  );

  return (
    <Tooltip content={`Cancel queue item #${itemId}`}>
      <IconButton
        aria-label={`Cancel queue item #${itemId}`}
        color="fg.muted"
        loading={busy}
        size="2xs"
        variant="ghost"
        onClick={onCancel}
      >
        <Icon as={XIcon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};
