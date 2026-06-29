import { HStack } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';

export const NotificationsHeaderActions = () => {
  const dispatch = useWorkbenchDispatch();
  const markAllRead = useCallback(() => dispatch({ type: 'markAllNotificationsRead' }), [dispatch]);
  const clearNotifications = useCallback(() => dispatch({ type: 'clearNotifications' }), [dispatch]);

  return (
    <HStack gap="2">
      <Button size="2xs" variant="outline" onClick={markAllRead}>
        Mark Read
      </Button>
      <Button size="2xs" variant="outline" onClick={clearNotifications}>
        Clear
      </Button>
    </HStack>
  );
};
