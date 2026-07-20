import type { WorkbenchNotificationKind } from '@workbench/projectContracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Badge, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { BellIcon, CircleCheckIcon, CircleXIcon, InfoIcon, TriangleAlertIcon, type LucideIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const kindColorPalette: Record<WorkbenchNotificationKind, string> = {
  error: 'red',
  info: 'blue',
  success: 'green',
};

const kindIcon = {
  error: CircleXIcon,
  info: InfoIcon,
  success: CircleCheckIcon,
} satisfies Record<WorkbenchNotificationKind, LucideIcon>;

export const NotificationsWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const { t } = useTranslation();
  const { errorCount, totalCount, unreadCount } = useWorkbenchSelector((snapshot) => ({
    errorCount: snapshot.notifications.filter((notification) => notification.kind === 'error').length,
    totalCount: snapshot.notifications.length,
    unreadCount: snapshot.notifications.filter((notification) => !notification.isRead).length,
  }));
  const label =
    unreadCount > 0
      ? t('notifications.newCount', { count: unreadCount })
      : t('notifications.totalCount', { count: totalCount });
  const icon = errorCount > 0 ? TriangleAlertIcon : BellIcon;

  if (region === 'bottom' && presentation !== 'expanded') {
    return <StatusWidgetChip icon={icon}>{t('notifications.labelWithCount', { label })}</StatusWidgetChip>;
  }

  return <NotificationsPanel />;
};

const NotificationsPanel = () => {
  const { t } = useTranslation();
  const notifications = useWorkbenchSelector((snapshot) => snapshot.notifications);

  return (
    <Stack flex="1" gap="3" minH="0" p="2">
      {notifications.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          {t('notifications.empty')}
        </Text>
      ) : (
        <Stack gap="2">
          {notifications.map((notification) => {
            const IconComponent = kindIcon[notification.kind];

            return (
              <Stack
                key={notification.id}
                bg={notification.isRead ? 'bg.subtle' : 'bg.muted'}
                borderWidth="1px"
                borderColor={notification.isRead ? 'border.subtle' : 'border.emphasized'}
                gap="1"
                p="2"
                rounded="md"
              >
                <HStack align="start" justify="space-between">
                  <HStack gap="2" minW="0">
                    <Icon as={IconComponent} color={`${kindColorPalette[notification.kind]}.300`} boxSize="3.5" />
                    <Text fontSize="2xs" fontWeight="700">
                      {notification.title}
                    </Text>
                  </HStack>
                  <Badge colorPalette={kindColorPalette[notification.kind]} size="xs">
                    {t(`notifications.kind.${notification.kind}`)}
                  </Badge>
                </HStack>
                {notification.message ? (
                  <Text color="fg.subtle" fontSize="2xs">
                    {notification.message}
                  </Text>
                ) : null}
                <Text color="fg.subtle" fontSize="2xs">
                  {notification.createdAt}
                </Text>
              </Stack>
            );
          })}
        </Stack>
      )}
    </Stack>
  );
};
