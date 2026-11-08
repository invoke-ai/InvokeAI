import type { MouseEvent as ReactMouseEvent } from 'react';

import { Icon } from '@chakra-ui/react';
import { describeQueueRetry } from '@features/queue/core/retryReport';
import { queueCommands } from '@features/queue/publicApi';
import { getApiErrorMessage } from '@platform/transport/http';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { RotateCcwIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { refreshQueue } from './queueDataStore';
import { useQueueUi } from './QueueUiContext';

/**
 * Re-queues a failed or canceled root item from its saved session; the backend reports which ids it took. A Canvas
 * job whose project is closed here is staged by another tab, so the confirmation says where its result goes.
 */
export const RetryQueueItemButton = ({
  isProjectClosedHere = false,
  itemId,
}: {
  isProjectClosedHere?: boolean;
  itemId: number;
}) => {
  const { t } = useTranslation();
  const { notify } = useQueueUi();
  const [busy, setBusy] = useState(false);

  const onRetry = useCallback(
    async (event: ReactMouseEvent) => {
      event.stopPropagation();
      setBusy(true);

      try {
        const retriedItemIds = await queueCommands.retryItems([itemId]);
        await refreshQueue();
        const report = describeQueueRetry({ isProjectClosedHere, itemIds: [itemId], retriedItemIds });

        notify[report.kind](t(report.titleKey), t(report.messageKey, report.values));
      } catch (error) {
        notify.error(t('widgets.queue.retryFailed'), getApiErrorMessage(error, t('widgets.queue.couldNotRetryItem')));
      } finally {
        setBusy(false);
      }
    },
    [isProjectClosedHere, itemId, notify, t]
  );

  return (
    <Tooltip content={t('widgets.queue.retryItem', { id: itemId })}>
      <IconButton
        aria-label={t('widgets.queue.retryItem', { id: itemId })}
        color="fg.muted"
        loading={busy}
        size="sm"
        variant="ghost"
        onClick={onRetry}
      >
        <Icon as={RotateCcwIcon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};
