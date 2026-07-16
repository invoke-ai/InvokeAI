import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { getUserScopedQueueCounts } from 'features/queue/store/userScopedQueueCounts';
import { useEffect } from 'react';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const baseTitle = document.title;
const invokeLogoSVG = 'assets/images/invoke-favicon.svg';
const invokeAlertLogoSVG = 'assets/images/invoke-alert-favicon.svg';

const queryOptions = {
  // The busy favicon/title reflect the user's own activity, not other users' generations.
  selectFromResult: (res) => {
    if (!res.data) {
      return { queueSize: 0 };
    }
    const { pending, inProgress } = getUserScopedQueueCounts(res.data.queue);
    return { queueSize: pending + inProgress };
  },
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
