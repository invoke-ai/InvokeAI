import { Button } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useEmptyModelCacheMutation } from 'services/api/endpoints/models';

const ClearModelCacheButton = () => {
  const [emptyModelCache, { isLoading }] = useEmptyModelCacheMutation();
  const { t } = useTranslation();

  const handleClearCache = useCallback(async () => {
    try {
      await emptyModelCache().unwrap();
      toast({
        status: 'success',
        title: t('modelCache.clearSucceeded'),
      });
    } catch {
      toast({
        status: 'error',
        title: t('modelCache.clearFailed'),
      });
    }
  }, [emptyModelCache, t]);

  return (
    <Button size="sm" colorScheme="red" onClick={handleClearCache} isLoading={isLoading}>
      {t('modelCache.clear')}
    </Button>
  );
};

export default memo(ClearModelCacheButton);
