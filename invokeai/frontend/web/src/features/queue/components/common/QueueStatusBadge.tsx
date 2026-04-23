import { Badge } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { SessionQueueItemStatus } from 'services/api/endpoints/queue';

export const QUEUE_STATUS_BADGE_STATES = {
  pending: { colorScheme: 'cyan', translationKey: 'queue.pending' },
  in_progress: { colorScheme: 'yellow', translationKey: 'queue.in_progress' },
  waiting: { colorScheme: 'purple', translationKey: 'queue.waiting' },
  completed: { colorScheme: 'green', translationKey: 'queue.completed' },
  failed: { colorScheme: 'red', translationKey: 'queue.failed' },
  canceled: { colorScheme: 'orange', translationKey: 'queue.canceled' },
};

export const getQueueStatusBadgeState = (status: SessionQueueItemStatus) => QUEUE_STATUS_BADGE_STATES[status];

const StatusBadge = ({ status }: { status: SessionQueueItemStatus }) => {
  const { t } = useTranslation();
  const badgeState = getQueueStatusBadgeState(status);
  return <Badge colorScheme={badgeState.colorScheme}>{t(badgeState.translationKey)}</Badge>;
};
export default memo(StatusBadge);
