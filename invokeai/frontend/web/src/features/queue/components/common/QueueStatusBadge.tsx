import { Badge } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { SessionQueueItemStatus } from 'services/api/endpoints/queue';

const STATUSES = {
  pending: { colorScheme: 'cyan', translationKey: 'queue.pending' },
  in_progress: { colorScheme: 'yellow', translationKey: 'queue.in_progress' },
  completed: { colorScheme: 'green', translationKey: 'queue.completed' },
  failed: { colorScheme: 'red', translationKey: 'queue.failed' },
  canceled: { colorScheme: 'orange', translationKey: 'queue.canceled' },
};

const StatusBadge = ({ status }: { status: SessionQueueItemStatus }) => {
  const { t } = useTranslation();
  return <Badge colorScheme={STATUSES[status].colorScheme}>{t(STATUSES[status].translationKey)}</Badge>;
};
export default memo(StatusBadge);
