import { getDeterminateProgressPercent } from '@features/queue/contracts';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useActiveQueueProgress } from '@workbench/queue-integration/useActiveQueueProgress';
import { useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { DOCUMENT_TITLE_BASE, formatDocumentTitle } from './documentTitle';

/**
 * Writes generation progress into the browser tab. Renders nothing — it is the
 * effect adapter for a piece of chrome React does not own.
 *
 * Percent is rounded before it reaches the title so a batch writes `document.title`
 * about a hundred times instead of once per progress frame.
 */
export const DocumentTitleProgress = () => {
  const { t } = useTranslation();
  const { progress, summary } = useActiveQueueProgress();
  const { current, total } = summary;
  const percent = getDeterminateProgressPercent(progress?.percentage);

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

  // Separate from the write above so switching titles does not flash the base
  // one in between; this only runs when the shell itself goes away.
  useMountEffect(() => () => {
    document.title = DOCUMENT_TITLE_BASE;
  });

  return null;
};
