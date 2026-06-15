import { HStack } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';

export const NotificationsHeaderActions = () => {
  const dispatch = useWorkbenchDispatch();

  return (
    <HStack gap="2">
      <Button size="2xs" variant="outline" onClick={() => dispatch({ type: 'markAllNotificationsRead' })}>
        Mark Read
      </Button>
      <Button size="2xs" variant="outline" onClick={() => dispatch({ type: 'clearNotifications' })}>
        Clear
      </Button>
    </HStack>
  );
};
