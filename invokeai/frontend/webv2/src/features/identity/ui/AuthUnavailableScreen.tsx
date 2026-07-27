import { Button } from '@platform/ui/Button';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { AuthScreen } from './AuthScreen';

/**
 * Fail-closed authentication availability boundary. This deliberately lives
 * outside the workbench shell so no account-owned provider mounts before a
 * retry has resolved the backend's auth mode and principal.
 */
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
