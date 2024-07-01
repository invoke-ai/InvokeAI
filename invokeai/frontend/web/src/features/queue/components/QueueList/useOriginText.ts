import { useTranslation } from 'react-i18next';
import type { SessionQueueItemDTO } from 'services/api/types';

export const useOriginText = (origin: SessionQueueItemDTO['origin']) => {
  const { t } = useTranslation();

  if (origin === 'canvas') {
    return t('queue.originCanvas');
  }

  if (origin === 'workflows') {
    return t('queue.originWorkflows');
  }

  return t('queue.originOther');
};
