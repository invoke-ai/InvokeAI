import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDisableInvocationCacheMutation, useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';
import { $isConnected } from 'services/events/stores';

export const useDisableInvocationCache = () => {
  const { t } = useTranslation();
  const { data: cacheStatus } = useGetInvocationCacheStatusQuery();
  const isConnected = useStore($isConnected);
  const [trigger, { isLoading }] = useDisableInvocationCacheMutation({
    fixedCacheKey: 'disableInvocationCache',
  });

  const isDisabled = useMemo(
    () => !cacheStatus?.enabled || !isConnected || cacheStatus?.max_size === 0,
    [cacheStatus?.enabled, cacheStatus?.max_size, isConnected]
  );

  const disableInvocationCache = useCallback(async () => {
    if (isDisabled) {
      return;
    }

    try {
      await trigger().unwrap();
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
  }, [isDisabled, trigger, t]);

  return { disableInvocationCache, isLoading, cacheStatus, isDisabled };
};
