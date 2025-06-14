import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useDisableInvocationCacheMutation, useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';
import { $isConnected } from 'services/events/stores';

export const useDisableInvocationCache = () => {
  const { t } = useTranslation();
  const { data: cacheStatus } = useGetInvocationCacheStatusQuery();
  const isConnected = useStore($isConnected);
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
    isDisabled: !cacheStatus?.enabled || !isConnected || cacheStatus?.max_size === 0,
  };
};
