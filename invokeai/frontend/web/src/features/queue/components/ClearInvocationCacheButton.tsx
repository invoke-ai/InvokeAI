import { Button } from '@invoke-ai/ui-library';
import { useClearInvocationCache } from 'features/queue/hooks/useClearInvocationCache';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ClearInvocationCacheButton = () => {
  const { t } = useTranslation();
  const { clearInvocationCache, isDisabled, isLoading } = useClearInvocationCache();

  return (
    <Button isDisabled={isDisabled} isLoading={isLoading} onClick={clearInvocationCache}>
      {t('invocationCache.clear')}
    </Button>
  );
};

export default memo(ClearInvocationCacheButton);
