import { HStack } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { useNotify } from '@workbench/useNotify';
import { useWorkbenchCommands, useWorkbenchQueries, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const NotificationsHeaderActions = () => {
  const { t } = useTranslation();
  const { notifications } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const notify = useNotify();
  const { hasAny, hasUnread } = useWorkbenchSelector((snapshot) => ({
    hasAny: snapshot.notifications.length > 0,
    hasUnread: snapshot.notifications.some((notification) => !notification.isRead),
  }));

  const markRead = useCallback(() => notifications.markAllRead(), [notifications]);
  // Clearing is reversible: the list is plain data, so Undo puts back what was there.
  const clear = useCallback(() => {
    const cleared = queries.getSnapshot().notifications;

    if (cleared.length === 0) {
      return;
    }

    notifications.clear();
    notify.undoable(t('notifications.cleared', { count: cleared.length }), {
      onUndo: () => notifications.restore(cleared),
      transient: true,
    });
  }, [notifications, notify, queries, t]);

  return (
    <HStack gap="2">
      <Button disabled={!hasUnread} size="sm" variant="outline" onClick={markRead}>
        {t('notifications.markRead')}
      </Button>
      <Button disabled={!hasAny} size="sm" variant="outline" onClick={clear}>
        {t('common.clear')}
      </Button>
    </HStack>
  );
};
