import type { BackendConnectionStatus } from '@platform/transport/types';
import type { WorkbenchNotificationCommands } from '@workbench/workbenchStore';

/** A dropout shorter than this passes unannounced, so a flapping socket does not spam notices. */
export const CONNECTION_NOTICE_DELAY_MS = 4_000;

/**
 * Announces an outage once, after it has lasted `delayMs`, and its end once, only if the outage was announced. The
 * first connection of a session is not news; nor is a reconnect nobody was told was needed.
 */
export const startConnectionNotices = ({
  delayMs = CONNECTION_NOTICE_DELAY_MS,
  getStatus,
  notifications,
  subscribe,
}: {
  delayMs?: number;
  getStatus: () => BackendConnectionStatus;
  notifications: Pick<WorkbenchNotificationCommands, 'add'>;
  subscribe: (listener: () => void) => () => void;
}): (() => void) => {
  let status = getStatus();
  let hasConnected = status === 'connected';
  let isAnnounced = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const announceOutage = () => {
    timer = undefined;
    isAnnounced = true;
    notifications.add({
      kind: 'info',
      message: 'Reconnecting… Queued work waits until the connection is back.',
      messageKey: 'notifications.connection.lostMessage',
      title: 'Lost connection to the server',
      titleKey: 'notifications.connection.lost',
    });
  };

  const unsubscribe = subscribe(() => {
    const next = getStatus();

    if (next === status) {
      return;
    }

    status = next;

    if (next === 'connected') {
      clearTimeout(timer);
      timer = undefined;
      hasConnected = true;

      if (isAnnounced) {
        isAnnounced = false;
        notifications.add({
          kind: 'info',
          title: 'Reconnected to the server',
          titleKey: 'notifications.connection.restored',
        });
      }
      return;
    }

    if (hasConnected && !isAnnounced && timer === undefined) {
      timer = setTimeout(announceOutage, delayMs);
    }
  });

  return () => {
    unsubscribe();
    clearTimeout(timer);
  };
};
