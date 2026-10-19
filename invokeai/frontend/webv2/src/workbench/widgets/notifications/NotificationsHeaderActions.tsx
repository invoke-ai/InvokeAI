import { HStack } from '@chakra-ui/react';
import { Button } from '@platform/ui';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useTranslation } from 'react-i18next';

export const NotificationsHeaderActions = () => {
  const { t } = useTranslation();
  const { notifications } = useWorkbenchCommands();

  return (
    <HStack gap="2">
      <Button size="sm" variant="outline" onClick={notifications.markAllRead}>
        {t('notifications.markRead')}
      </Button>
      <Button size="sm" variant="outline" onClick={notifications.clear}>
        {t('common.clear')}
      </Button>
    </HStack>
  );
};
