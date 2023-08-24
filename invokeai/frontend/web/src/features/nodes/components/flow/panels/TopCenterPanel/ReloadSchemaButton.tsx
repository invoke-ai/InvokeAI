import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
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
    <IAIIconButton
      icon={<FaSyncAlt />}
      tooltip={t('nodes.reloadSchema')}
      aria-label={t('nodes.reloadSchema')}
      onClick={handleReloadSchema}
    />
  );
};

export default memo(ReloadSchemaButton);
