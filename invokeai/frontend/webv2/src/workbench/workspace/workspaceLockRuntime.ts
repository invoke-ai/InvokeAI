import type { WorkbenchLayoutCommands, WorkbenchNotificationCommands } from '@workbench/workbenchStore';

import {
  getWorkbenchSettingsSnapshot,
  patchWorkbenchPreferences,
  subscribeWorkbenchSettings,
} from '@workbench/settings/store';
import i18next from 'i18next';

const TITLE_KEY = 'shell.workspace.lockOfferTitle';
const MESSAGE_KEY = 'shell.workspace.lockOfferMessage';

/**
 * The editor's side of "Rearrange without Customize". An account that kept rearranging after the upgrade is offered the
 * lock once: as soon as its settings are ready, the offer is recorded and marked shown in the same step. Turning the
 * setting off clears the layout history outside Customize, so a locked undo cannot bring back a structural change.
 */
export const startWorkspaceLock = (commands: {
  layout: Pick<WorkbenchLayoutCommands, 'clearHistory'>;
  notifications: Pick<WorkbenchNotificationCommands, 'add'>;
}): (() => void) => {
  let wasRearranging = getWorkbenchSettingsSnapshot().preferences.rearrangeWithoutCustomize;

  const update = () => {
    const { preferences, status } = getWorkbenchSettingsSnapshot();

    if (wasRearranging && !preferences.rearrangeWithoutCustomize) {
      commands.layout.clearHistory();
    }
    wasRearranging = preferences.rearrangeWithoutCustomize;

    if (status !== 'ready' || !preferences.rearrangeWithoutCustomize || preferences.workspaceLockOfferShown) {
      return;
    }

    // Updates the snapshot before it returns, so the notification below cannot be recorded twice.
    void patchWorkbenchPreferences({ workspaceLockOfferShown: true });
    commands.notifications.add({
      actions: [{ kind: 'lock-workspaces' }, { kind: 'keep-rearranging' }],
      kind: 'info',
      message: i18next.t(MESSAGE_KEY, { lng: 'en' }),
      messageKey: MESSAGE_KEY,
      title: i18next.t(TITLE_KEY, { lng: 'en' }),
      titleKey: TITLE_KEY,
    });
  };

  update();

  return subscribeWorkbenchSettings(update);
};
