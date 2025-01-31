import { Button } from '@invoke-ai/ui-library';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useEmptyModelCacheMutation } from 'services/api/endpoints/models';

const ClearModelCacheButton = () => {
  const isModelCacheEnabled = useFeatureStatus('modelCache');
  const [emptyModelCache, { isLoading }] = useEmptyModelCacheMutation();
  const { t } = useTranslation();

  const handleClearCache = useCallback(async () => {
    try {
      await emptyModelCache().unwrap();
      toast({
        status: 'success',
        title: t('modelCache.clearSucceeded'),
      });
    } catch (error) {
      toast({
        status: 'error',
        title: t('modelCache.clearFailed'),
      });
    }
  }, [emptyModelCache, t]);

  if (!isModelCacheEnabled) {
    return <></>;
  }

  return (
    <Button size="sm" colorScheme="red" onClick={handleClearCache} isLoading={isLoading}>
      {t('modelCache.clear')}
    </Button>
  );
};

export default memo(ClearModelCacheButton);
