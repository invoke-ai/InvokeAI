import { watchSessionExpiry } from '@features/identity/session';
import { useMountEffect } from '@platform/react/useMountEffect';
import { createActionToast } from '@platform/ui/toaster';
import i18next from 'i18next';

/**
 * Warns a few minutes before the signed-in session would end. The backend renews it on every change the user makes,
 * so only an idle session gets this close; the warning closes itself once a renewal or sign-out makes it moot.
 */
export const SessionExpiryNotice = () => {
  useMountEffect(() =>
    watchSessionExpiry(({ expiresAt, signal }) => {
      createActionToast({
        actions: [],
        description: i18next.t('auth.sessionEndingDescription', {
          time: new Intl.DateTimeFormat(i18next.resolvedLanguage, { timeStyle: 'short' }).format(expiresAt),
        }),
        duration: Number.POSITIVE_INFINITY,
        signal,
        title: i18next.t('auth.sessionEndingTitle'),
        type: 'warning',
      });
    })
  );

  return null;
};
