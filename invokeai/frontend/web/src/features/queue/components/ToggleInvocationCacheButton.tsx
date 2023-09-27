import IAIButton from 'common/components/IAIButton';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';
import { useDisableInvocationCache } from '../hooks/useDisableInvocationCache';
import { useEnableInvocationCache } from '../hooks/useEnableInvocationCache';

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
      <IAIButton
        isDisabled={isDisableDisabled}
        isLoading={isDisableLoading}
        onClick={disableInvocationCache}
      >
        {t('invocationCache.disable')}
      </IAIButton>
    );
  }

  return (
    <IAIButton
      isDisabled={isEnableDisabled}
      isLoading={isEnableLoading}
      onClick={enableInvocationCache}
    >
      {t('invocationCache.enable')}
    </IAIButton>
  );
};

export default memo(ToggleInvocationCacheButton);
