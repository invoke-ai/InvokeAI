import { useTranslation } from 'react-i18next';
import type { S } from 'services/api/types';

export const useOriginText = (origin: S['SessionQueueItem']['origin']) => {
  const { t } = useTranslation();

  if (origin === 'generation') {
    return t('queue.generation');
  }

  if (origin === 'workflows') {
    return t('queue.workflows');
  }

  if (origin === 'upscaling') {
    return t('queue.upscaling');
  }

  return t('queue.other');
};
