import { useWorkbenchSettingsSelector } from '@workbench/settings/store';

import { useWhatsNewSnapshot } from './whatsNewStore';

export interface WhatsNewState {
  isOpen: boolean;
  /** Whether this showing is the automatic one for a version the account has not dismissed yet. */
  isUnseen: boolean;
  /** The server version, or null while unknown. */
  version: string | null;
}

/** Notes once per server version and account, after preferences load and the beta notice, or on request. */
export const useWhatsNew = (): WhatsNewState => {
  const { isDismissed, isRequested, version } = useWhatsNewSnapshot();
  const isUnseen = useWorkbenchSettingsSelector(
    (snapshot) =>
      !isDismissed &&
      version !== null &&
      snapshot.status === 'ready' &&
      snapshot.preferences.betaNoticeAcknowledged &&
      snapshot.preferences.whatsNewSeenVersion !== version
  );

  return { isOpen: isRequested || isUnseen, isUnseen, version };
};
