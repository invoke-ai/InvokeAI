import { HStack } from '@chakra-ui/react';

import { Button } from '../../components/ui/Button';
import { useWorkbenchDispatch } from '../../WorkbenchContext';

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
