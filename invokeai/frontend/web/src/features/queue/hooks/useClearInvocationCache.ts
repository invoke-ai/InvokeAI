import { useStore } from '@nanostores/react';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useClearInvocationCacheMutation, useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';
import { $isConnected } from 'services/events/stores';

export const useClearInvocationCache = () => {
  const { t } = useTranslation();
  const { data: cacheStatus } = useGetInvocationCacheStatusQuery();
  const isConnected = useStore($isConnected);
  const [trigger, { isLoading }] = useClearInvocationCacheMutation({
    fixedCacheKey: 'clearInvocationCache',
  });

  const isDisabled = useMemo(() => !cacheStatus?.size || !isConnected, [cacheStatus?.size, isConnected]);

  const clearInvocationCache = useCallback(async () => {
    if (isDisabled) {
      return;
    }

    try {
      await trigger().unwrap();
      toast({
        id: 'INVOCATION_CACHE_CLEAR_SUCCEEDED',
        title: t('invocationCache.clearSucceeded'),
        status: 'success',
      });
    } catch {
      toast({
        id: 'INVOCATION_CACHE_CLEAR_FAILED',
        title: t('invocationCache.clearFailed'),
        status: 'error',
      });
    }
  }, [isDisabled, trigger, t]);

  return { clearInvocationCache, isLoading, cacheStatus, isDisabled };
};
