import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useEffect } from 'react';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const baseTitle = document.title;
const invokeLogoSVG = 'assets/images/invoke-favicon.svg';
const invokeAlertLogoSVG = 'assets/images/invoke-alert-favicon.svg';

const queryOptions = {
  selectFromResult: (res) => ({
    queueSize: res.data ? res.data.queue.pending + res.data.queue.in_progress : 0,
  }),
} satisfies Parameters<typeof useGetQueueStatusQuery>[1];

const updateFavicon = (queueSize: number) => {
  document.title = queueSize > 0 ? `(${queueSize}) ${baseTitle}` : baseTitle;
  const faviconEl = document.getElementById('invoke-favicon');
  if (faviconEl instanceof HTMLLinkElement) {
    faviconEl.href = queueSize > 0 ? invokeAlertLogoSVG : invokeLogoSVG;
  }
};

/**
 * This hook synchronizes the queue status with the page's title and favicon.
 * It should be considered a singleton and only used once in the component tree.
 */
export const useSyncFaviconQueueStatus = () => {
  useAssertSingleton('useSyncFaviconQueueStatus');
  const { queueSize } = useGetQueueStatusQuery(undefined, queryOptions);
  useEffect(() => {
    updateFavicon(queueSize);
  }, [queueSize]);
};
