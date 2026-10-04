import { useStore } from '@nanostores/react';
import { useIsAdmin } from 'features/auth/hooks/useIsAdmin';
import { getIsEnableInvocationCacheDisabled } from 'features/queue/hooks/invocationCacheControls';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useEnableInvocationCacheMutation, useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';
import { $isConnected } from 'services/events/stores';

export const useEnableInvocationCache = () => {
  const { t } = useTranslation();
  const { data: cacheStatus } = useGetInvocationCacheStatusQuery();
  const isConnected = useStore($isConnected);
  const isAdmin = useIsAdmin();
  const [_trigger, { isLoading }] = useEnableInvocationCacheMutation({
    fixedCacheKey: 'enableInvocationCache',
  });

  const trigger = useCallback(async () => {
    try {
      await _trigger().unwrap();
      toast({
        id: 'INVOCATION_CACHE_ENABLE_SUCCEEDED',
        title: t('invocationCache.enableSucceeded'),
        status: 'success',
      });
    } catch {
      toast({
        id: 'INVOCATION_CACHE_ENABLE_FAILED',
        title: t('invocationCache.enableFailed'),
        status: 'error',
      });
    }
  }, [_trigger, t]);

  return {
    trigger,
    isLoading,
    isDisabled: getIsEnableInvocationCacheDisabled(isAdmin, isConnected, cacheStatus),
  };
};
