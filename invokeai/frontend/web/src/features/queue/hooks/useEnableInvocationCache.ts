import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useEnableInvocationCacheMutation, useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';

export const useEnableInvocationCache = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isCacheEnabled = useFeatureStatus('invocationCache').isFeatureEnabled;
  const { data: cacheStatus } = useGetInvocationCacheStatusQuery(undefined, { skip: isCacheEnabled });
  const isConnected = useAppSelector((s) => s.system.isConnected);
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
      dispatch(
        addToast({
          title: t('invocationCache.enableSucceeded'),
          status: 'success',
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('invocationCache.enableFailed'),
          status: 'error',
        })
      );
    }
  }, [isDisabled, trigger, dispatch, t]);

  return { enableInvocationCache, isLoading, cacheStatus, isDisabled };
};
