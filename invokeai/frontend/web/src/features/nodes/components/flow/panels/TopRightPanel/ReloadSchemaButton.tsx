import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsClockwiseBold } from 'react-icons/pi'
import { receivedOpenAPISchema } from 'services/api/thunks/schema';

const ReloadNodeTemplatesButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleReloadSchema = useCallback(() => {
    dispatch(receivedOpenAPISchema());
  }, [dispatch]);

  return (
    <InvButton
      leftIcon={<PiArrowsClockwiseBold />}
      tooltip={t('nodes.reloadNodeTemplates')}
      aria-label={t('nodes.reloadNodeTemplates')}
      onClick={handleReloadSchema}
    >
      {t('nodes.reloadNodeTemplates')}
    </InvButton>
  );
};

export default memo(ReloadNodeTemplatesButton);
