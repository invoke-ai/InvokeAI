import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery, usePauseProcessorMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const usePauseProcessor = () => {
  const { t } = useTranslation();
  const isConnected = useStore($isConnected);
  const { data: queueStatus } = useGetQueueStatusQuery();
  const currentUser = useAppSelector(selectCurrentUser);
  const [_trigger, { isLoading }] = usePauseProcessorMutation({
    fixedCacheKey: 'pauseProcessor',
  });

  // Only admin users can pause the processor
  const isAdmin = useMemo(() => currentUser?.is_admin ?? false, [currentUser]);

  const trigger = useCallback(async () => {
    try {
      await _trigger().unwrap();
      toast({
        id: 'PAUSE_SUCCEEDED',
        title: t('queue.pauseSucceeded'),
        status: 'success',
      });
    } catch {
      toast({
        id: 'PAUSE_FAILED',
        title: t('queue.pauseFailed'),
        status: 'error',
      });
    }
  }, [_trigger, t]);

  return { trigger, isLoading, isDisabled: !isConnected || !queueStatus?.processor.is_started || !isAdmin };
};
