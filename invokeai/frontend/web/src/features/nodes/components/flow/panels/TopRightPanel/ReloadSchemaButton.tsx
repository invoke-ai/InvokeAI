import { Button } from '@invoke-ai/ui-library';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsClockwiseBold } from 'react-icons/pi';
import { useLazyLoadSchemaQuery } from 'services/api/endpoints/utilities';

const ReloadNodeTemplatesButton = () => {
  const { t } = useTranslation();
  const [_loadSchema] = useLazyLoadSchemaQuery();

  const handleReloadSchema = useCallback(() => {
    _loadSchema();
  }, [_loadSchema]);

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
