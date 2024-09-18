import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useEnableInvocationCacheMutation, useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';
import { $isConnected } from 'services/events/stores';

export const useEnableInvocationCache = () => {
  const { t } = useTranslation();
  const { data: cacheStatus } = useGetInvocationCacheStatusQuery();
  const isConnected = useStore($isConnected);
  const [trigger, { isLoading }] = useEnableInvocationCacheMutation({
    fixedCacheKey: 'enableInvocationCache',
  });

  const isDisabled = useMemo(
    () => cacheStatus?.enabled || !isConnected || cacheStatus?.max_size === 0,
    [cacheStatus?.enabled, cacheStatus?.max_size, isConnected]
  );

  const enableInvocationCache = useCallback(async () => {
    if (isDisabled) {
      return;
    }

    try {
      await trigger().unwrap();
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
  }, [isDisabled, trigger, t]);

  return { enableInvocationCache, isLoading, cacheStatus, isDisabled };
};
