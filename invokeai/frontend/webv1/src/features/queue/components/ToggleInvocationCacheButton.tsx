import { Button } from '@invoke-ai/ui-library';
import { useDisableInvocationCache } from 'features/queue/hooks/useDisableInvocationCache';
import { useEnableInvocationCache } from 'features/queue/hooks/useEnableInvocationCache';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';

const ToggleInvocationCacheButton = () => {
  const { t } = useTranslation();
  const { data: cacheStatus } = useGetInvocationCacheStatusQuery();

  const enableInvocationCache = useEnableInvocationCache();

  const disableInvocationCache = useDisableInvocationCache();

  if (cacheStatus?.enabled) {
    return (
      <Button
        onClick={disableInvocationCache.trigger}
        isDisabled={disableInvocationCache.isDisabled}
        isLoading={disableInvocationCache.isLoading}
      >
        {t('invocationCache.disable')}
      </Button>
    );
  }

  return (
    <Button
      onClick={enableInvocationCache.trigger}
      isDisabled={enableInvocationCache.isDisabled}
      isLoading={enableInvocationCache.isLoading}
    >
      {t('invocationCache.enable')}
    </Button>
  );
};

export default memo(ToggleInvocationCacheButton);
