import { useStore } from '@nanostores/react';
import { useIsAdmin } from 'features/auth/hooks/useIsAdmin';
import { getIsDisableInvocationCacheDisabled } from 'features/queue/hooks/invocationCacheControls';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useDisableInvocationCacheMutation, useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';
import { $isConnected } from 'services/events/stores';

export const useDisableInvocationCache = () => {
  const { t } = useTranslation();
  const { data: cacheStatus } = useGetInvocationCacheStatusQuery();
  const isConnected = useStore($isConnected);
  const isAdmin = useIsAdmin();
  const [_trigger, { isLoading }] = useDisableInvocationCacheMutation({
    fixedCacheKey: 'disableInvocationCache',
  });

  const trigger = useCallback(async () => {
    try {
      await _trigger().unwrap();
      toast({
        id: 'INVOCATION_CACHE_DISABLE_SUCCEEDED',
        title: t('invocationCache.disableSucceeded'),
        status: 'success',
      });
    } catch {
      toast({
        id: 'INVOCATION_CACHE_DISABLE_FAILED',
        title: t('invocationCache.disableFailed'),
        status: 'error',
      });
    }
  }, [_trigger, t]);

  return {
    trigger,
    isLoading,
    cacheStatus,
    isDisabled: getIsDisableInvocationCacheDisabled(isAdmin, isConnected, cacheStatus),
  };
};
