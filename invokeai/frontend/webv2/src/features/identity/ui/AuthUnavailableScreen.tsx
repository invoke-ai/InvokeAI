import { Button } from '@platform/ui/Button';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { AuthScreen } from './AuthScreen';

/** Keep account-owned providers unmounted until auth mode and principal resolve successfully. */
export const AuthUnavailableScreen = ({ onRetry }: { onRetry: () => Promise<void> | void }) => {
  const { t } = useTranslation();
  const handleRetry = useCallback(() => {
    void onRetry();
  }, [onRetry]);

  return (
    <AuthScreen subtitle={t('widgets.serverStatus.disconnected')} title={t('common.somethingWentWrong')}>
      <Button onClick={handleRetry}>{t('common.retry')}</Button>
    </AuthScreen>
  );
};
