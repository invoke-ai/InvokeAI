import { Button } from '@invoke-ai/ui-library';
import { useDisableInvocationCache } from 'features/queue/hooks/useDisableInvocationCache';
import { useEnableInvocationCache } from 'features/queue/hooks/useEnableInvocationCache';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';

const ToggleInvocationCacheButton = () => {
  const { t } = useTranslation();
  const { data: cacheStatus } = useGetInvocationCacheStatusQuery();

  const {
    enableInvocationCache,
    isDisabled: isEnableDisabled,
    isLoading: isEnableLoading,
  } = useEnableInvocationCache();

  const {
    disableInvocationCache,
    isDisabled: isDisableDisabled,
    isLoading: isDisableLoading,
  } = useDisableInvocationCache();

  if (cacheStatus?.enabled) {
    return (
      <Button isDisabled={isDisableDisabled} isLoading={isDisableLoading} onClick={disableInvocationCache}>
        {t('invocationCache.disable')}
      </Button>
    );
  }

  return (
    <Button isDisabled={isEnableDisabled} isLoading={isEnableLoading} onClick={enableInvocationCache}>
      {t('invocationCache.enable')}
    </Button>
  );
};

export default memo(ToggleInvocationCacheButton);
