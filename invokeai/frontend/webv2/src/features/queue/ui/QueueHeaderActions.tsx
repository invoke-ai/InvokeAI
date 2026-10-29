import { Icon } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { XIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { getQueueAction, useQueueActions } from './queueMenuActions';

/** Cancelling the running item is cheap (Retry brings it back), so the header button acts without asking. */
export const QueueHeaderActions = () => {
  const { t } = useTranslation();
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
    <Button disabled={cancelCurrent?.disabled ?? true} loading={busy} size="sm" variant="outline" onClick={onCancel}>
      <Icon as={XIcon} boxSize="3.5" />
      {t('widgets.queue.cancelCurrent')}
    </Button>
  );
};
