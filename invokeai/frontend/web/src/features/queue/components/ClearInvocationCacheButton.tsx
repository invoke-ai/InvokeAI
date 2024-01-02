import { InvButton } from 'common/components/InvButton/InvButton';
import { useClearInvocationCache } from 'features/queue/hooks/useClearInvocationCache';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ClearInvocationCacheButton = () => {
  const { t } = useTranslation();
  const { clearInvocationCache, isDisabled, isLoading } =
    useClearInvocationCache();

  return (
    <InvButton
      isDisabled={isDisabled}
      isLoading={isLoading}
      onClick={clearInvocationCache}
    >
      {t('invocationCache.clear')}
    </InvButton>
  );
};

export default memo(ClearInvocationCacheButton);
