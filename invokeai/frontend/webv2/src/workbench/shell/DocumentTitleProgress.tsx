import {
  getDeterminateProgressPercent,
  getFinishedImageCount,
  getQueueItemExpectedImageCount,
} from '@features/queue/contracts';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useActiveQueueProgress } from '@workbench/queue-integration/useActiveQueueProgress';
import { useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { DOCUMENT_TITLE_BASE, formatDocumentTitle, getBatchDocumentTitlePercent } from './documentTitle';

/** Write rounded progress to document.title to avoid updating on every progress frame. */
export const DocumentTitleProgress = () => {
  const { t } = useTranslation();
  const { isConcurrentProgress, progress, queueItems, summary } = useActiveQueueProgress();
  let { current, total } = summary;
  let percent = getDeterminateProgressPercent(progress?.percentage);
  const runningItem = queueItems.find((item) => item.id === summary.runningQueueItemId);
  if (isConcurrentProgress && runningItem) {
    // Concurrent progress describes the active batch, not whichever GPU last reported or the entire queue.
    const batchCount = getQueueItemExpectedImageCount(runningItem);
    const settledCount = getFinishedImageCount(runningItem, progress);
    percent = getBatchDocumentTitlePercent({ batchCount, completedCount: settledCount });
    current = Math.min(batchCount, settledCount + 1);
    total = batchCount;
  }

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
