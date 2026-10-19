import { useWorkbenchSettingsSelector } from '@workbench/settings/store';

import { useWhatsNewSnapshot } from './whatsNewStore';

export interface WhatsNewState {
  isOpen: boolean;
  /** Whether this showing is the automatic one for a version the account has not dismissed yet. */
  isUnseen: boolean;
  /** The server version, or null while unknown. */
  version: string | null;
}

/**
 * Show the notes once per server version for each account, after its preferences are known and the alpha
 * notice is out of the way, and whenever the app menu asks for them.
 */
export const useWhatsNew = (): WhatsNewState => {
  const { isDismissed, isRequested, version } = useWhatsNewSnapshot();
  const isUnseen = useWorkbenchSettingsSelector(
    (snapshot) =>
      !isDismissed &&
      version !== null &&
      snapshot.status === 'ready' &&
      snapshot.preferences.alphaNoticeAcknowledged &&
      snapshot.preferences.whatsNewSeenVersion !== version
  );

  return { isOpen: isRequested || isUnseen, isUnseen, version };
};
