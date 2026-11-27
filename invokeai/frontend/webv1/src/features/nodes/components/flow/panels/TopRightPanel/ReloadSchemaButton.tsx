import { Button } from '@invoke-ai/ui-library';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsClockwiseBold } from 'react-icons/pi';
import { useLazyGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

const ReloadNodeTemplatesButton = () => {
  const { t } = useTranslation();
  const [_getOpenAPISchema] = useLazyGetOpenAPISchemaQuery();

  const handleReloadSchema = useCallback(() => {
    _getOpenAPISchema();
  }, [_getOpenAPISchema]);

  return (
    <Button
      leftIcon={<PiArrowsClockwiseBold />}
      tooltip={t('nodes.reloadNodeTemplates')}
      aria-label={t('nodes.reloadNodeTemplates')}
      onClick={handleReloadSchema}
    >
      {t('nodes.reloadNodeTemplates')}
    </Button>
  );
};

export default memo(ReloadNodeTemplatesButton);
