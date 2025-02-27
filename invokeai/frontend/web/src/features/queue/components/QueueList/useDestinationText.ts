import { useTranslation } from 'react-i18next';
import type { SessionQueueItemDTO } from 'services/api/types';

export const useDestinationText = (destination: SessionQueueItemDTO['destination']) => {
  const { t } = useTranslation();

  if (destination === 'canvas') {
    return t('queue.canvas');
  }

  if (destination === 'gallery') {
    return t('queue.gallery');
  }

  return t('queue.other');
};
