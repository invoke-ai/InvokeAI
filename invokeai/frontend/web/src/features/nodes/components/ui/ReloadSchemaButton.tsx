import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { BiRefresh } from 'react-icons/bi';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';

export default function ReloadSchemaButton() {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleReloadSchema = useCallback(() => {
    dispatch(receivedOpenAPISchema());
  }, [dispatch]);

  return (
    <IAIIconButton
      icon={<BiRefresh />}
      fontSize={24}
      tooltip={t('nodes.reloadSchema')}
      aria-label={t('nodes.reloadSchema')}
      onClick={handleReloadSchema}
    />
  );
}
