import type { MouseEvent as ReactMouseEvent } from 'react';

import { Icon } from '@chakra-ui/react';
import { queueCommands } from '@features/queue/publicApi';
import { getApiErrorMessage } from '@platform/transport/http';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { XIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { refreshQueue } from './queueDataStore';
import { useQueueUi } from './QueueUiContext';

/** Cancel by backend ID across clients; local coordination follows status events. */
export const CancelQueueItemButton = ({ itemId }: { itemId: number }) => {
  const { t } = useTranslation();
  const { notify } = useQueueUi();
  const [busy, setBusy] = useState(false);

  const onCancel = useCallback(
    async (event: ReactMouseEvent) => {
      event.stopPropagation();
      setBusy(true);

      try {
        await queueCommands.cancelItem(itemId);
        await refreshQueue();
      } catch (error) {
        notify.error(t('common.cancelFailed'), getApiErrorMessage(error, t('widgets.queue.couldNotCancelItem')));
      } finally {
        setBusy(false);
      }
    },
    [itemId, notify, t]
  );

  return (
    <Tooltip content={t('widgets.queue.cancelItem', { id: itemId })}>
      <IconButton
        aria-label={t('widgets.queue.cancelItem', { id: itemId })}
        color="fg.muted"
        loading={busy}
        size="sm"
        variant="ghost"
        onClick={onCancel}
      >
        <Icon as={XIcon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};
