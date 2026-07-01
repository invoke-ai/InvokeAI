import { HStack } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const NotificationsHeaderActions = () => {
  const { t } = useTranslation();
  const dispatch = useWorkbenchDispatch();
  const markAllRead = useCallback(() => dispatch({ type: 'markAllNotificationsRead' }), [dispatch]);
  const clearNotifications = useCallback(() => dispatch({ type: 'clearNotifications' }), [dispatch]);

  return (
    <HStack gap="2">
      <Button size="2xs" variant="outline" onClick={markAllRead}>
        {t('notifications.markRead')}
      </Button>
      <Button size="2xs" variant="outline" onClick={clearNotifications}>
        {t('common.clear')}
      </Button>
    </HStack>
  );
};
