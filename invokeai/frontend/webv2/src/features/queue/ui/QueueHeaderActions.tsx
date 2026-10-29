import { Icon } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { XIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { getQueueAction, useQueueActions } from './queueMenuActions';
import { useQueueUi } from './QueueUiContext';

/**
 * Cancelling the running item is cheap (Retry brings it back), so the header button acts without asking. Offline it
 * says why it cannot act: the server's running item is a stale reading until the connection returns.
 */
export const QueueHeaderActions = () => {
  const { t } = useTranslation();
  const { isConnected } = useQueueUi();
  const cancelCurrent = getQueueAction(useQueueActions(), 'cancel-current');
  const [busy, setBusy] = useState(false);

  const onCancel = useCallback(async () => {
    if (!cancelCurrent || cancelCurrent.disabled) {
      return;
    }

    setBusy(true);
    try {
      await cancelCurrent.trigger();
    } finally {
      setBusy(false);
    }
  }, [cancelCurrent]);

  return (
    <Button
      disabled={cancelCurrent?.disabled ?? true}
      loading={busy}
      size="sm"
      unavailableReason={isConnected ? undefined : t('widgets.queue.unavailableWhileDisconnected')}
      variant="outline"
      onClick={onCancel}
    >
      <Icon as={XIcon} boxSize="3.5" />
      {t('widgets.queue.cancelCurrent')}
    </Button>
  );
};
