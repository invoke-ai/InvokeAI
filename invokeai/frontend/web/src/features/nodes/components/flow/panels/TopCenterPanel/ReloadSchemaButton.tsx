import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSyncAlt } from 'react-icons/fa';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';

const ReloadSchemaButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleReloadSchema = useCallback(() => {
    dispatch(receivedOpenAPISchema());
  }, [dispatch]);

  return (
    <IAIButton
      leftIcon={<FaSyncAlt />}
      tooltip={t('nodes.reloadSchema')}
      aria-label={t('nodes.reloadSchema')}
      onClick={handleReloadSchema}
    >
      {t('nodes.reloadSchema')}
    </IAIButton>
  );
};

export default memo(ReloadSchemaButton);
