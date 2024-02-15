import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDisableInvocationCacheMutation, useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';

export const useDisableInvocationCache = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isCacheEnabled = useFeatureStatus('invocationCache').isFeatureEnabled;
  const { data: cacheStatus } = useGetInvocationCacheStatusQuery(undefined, { skip: isCacheEnabled });
  const isConnected = useAppSelector((s) => s.system.isConnected);
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
      dispatch(
        addToast({
          title: t('invocationCache.disableSucceeded'),
          status: 'success',
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('invocationCache.disableFailed'),
          status: 'error',
        })
      );
    }
  }, [isDisabled, trigger, dispatch, t]);

  return { disableInvocationCache, isLoading, cacheStatus, isDisabled };
};
