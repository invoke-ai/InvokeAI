import { getDeterminateProgressPercent, getQueueItemExpectedImageCount } from '@features/queue/contracts';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useActiveQueueProgress } from '@workbench/queue-integration/useActiveQueueProgress';
import { useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { DOCUMENT_TITLE_BASE, formatDocumentTitle, getBatchDocumentTitlePercent } from './documentTitle';

/** Write rounded progress to document.title to avoid updating on every progress frame. */
export const DocumentTitleProgress = () => {
  const { t } = useTranslation();
  const { progress, queueItems, summary } = useActiveQueueProgress();
  const { current, runningQueueItemId, total } = summary;
  const runningItem = queueItems.find((item) => item.id === runningQueueItemId);
  const batchCount = runningItem ? getQueueItemExpectedImageCount(runningItem) : 1;
  // A multi-image Invoke can run on several GPUs. The most recent worker's
  // percentage isn't batch progress; settled slots provide a stable ratio.
  const settledCount =
    (runningItem?.completedBackendItemIds?.length ?? 0) + (runningItem?.cancelledBackendItemIds?.length ?? 0);
  const percent = getBatchDocumentTitlePercent({
    batchCount,
    completedCount: Math.max(progress?.completedItemCount ?? 0, settledCount),
    singleItemPercent: getDeterminateProgressPercent(progress?.percentage),
  });

  const title = useMemo(
    () =>
      formatDocumentTitle({
        current,
        labels: { generating: t('common.generating'), queued: t('widgets.queueStatus.queued', { count: total }) },
        percent,
        total,
      }),
    [current, percent, t, total]
  );

  useEffect(() => {
    document.title = title;
  }, [title]);

  // Separate from the write above so switching titles does not flash the base one in between; this only runs when the
  // shell itself goes away.
  useMountEffect(() => () => {
    document.title = DOCUMENT_TITLE_BASE;
  });

  return null;
};
