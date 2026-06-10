import { Button, HStack } from '@chakra-ui/react';

import { useWorkbench } from '../../WorkbenchContext';

export const NotificationsHeaderActions = () => {
  const { dispatch } = useWorkbench();

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
