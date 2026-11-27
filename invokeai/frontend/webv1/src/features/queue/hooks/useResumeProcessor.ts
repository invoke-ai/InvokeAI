import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import { useGetQueueStatusQuery, useResumeProcessorMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useResumeProcessor = () => {
  const isConnected = useStore($isConnected);
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { t } = useTranslation();
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();
  const [_trigger, { isLoading }] = useResumeProcessorMutation({
    fixedCacheKey: 'resumeProcessor',
  });

  // In single-user mode, treat as admin. In multiuser mode, check is_admin flag.
  const isAdmin = useMemo(() => {
    if (setupStatus && !setupStatus.multiuser_enabled) {
      return true;
    }
    return currentUser?.is_admin ?? false;
  }, [setupStatus, currentUser]);

  const trigger = useCallback(async () => {
    try {
      await _trigger().unwrap();
      toast({
        id: 'PROCESSOR_RESUMED',
        title: t('queue.resumeSucceeded'),
        status: 'success',
      });
    } catch {
      toast({
        id: 'PROCESSOR_RESUME_FAILED',
        title: t('queue.resumeFailed'),
        status: 'error',
      });
    }
  }, [_trigger, t]);

  return { trigger, isLoading, isDisabled: !isConnected || queueStatus?.processor.is_started || !isAdmin };
};
