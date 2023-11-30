import IAIButton from 'common/components/IAIButton';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useClearInvocationCache } from 'features/queue/hooks/useClearInvocationCache';

const ClearInvocationCacheButton = () => {
  const { t } = useTranslation();
  const { clearInvocationCache, isDisabled, isLoading } =
    useClearInvocationCache();

  return (
    <IAIButton
      isDisabled={isDisabled}
      isLoading={isLoading}
      onClick={clearInvocationCache}
    >
      {t('invocationCache.clear')}
    </IAIButton>
  );
};

export default memo(ClearInvocationCacheButton);
