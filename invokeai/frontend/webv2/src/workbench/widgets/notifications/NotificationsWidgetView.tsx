import { Badge, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { BellIcon, CircleCheckIcon, CircleXIcon, InfoIcon, TriangleAlertIcon, type LucideIcon } from 'lucide-react';

import { StatusWidgetChip } from '../../components/WidgetFrames';
import type { WidgetViewProps, WorkbenchNotificationKind } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';

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
  const { state } = useWorkbench();
  const unreadCount = state.notifications.filter((notification) => !notification.isRead).length;
  const errorCount = state.notifications.filter((notification) => notification.kind === 'error').length;
  const label = unreadCount > 0 ? `${unreadCount} new` : `${state.notifications.length} total`;
  const icon = errorCount > 0 ? TriangleAlertIcon : BellIcon;

  if (region === 'bottom' && presentation !== 'expanded') {
    return <StatusWidgetChip icon={icon}>Notifications: {label}</StatusWidgetChip>;
  }

  return <NotificationsPanel />;
};

const NotificationsPanel = () => {
  const { state } = useWorkbench();

  return (
    <Stack flex="1" gap="3" minH="0">
      {state.notifications.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          Successful operations, errors, and system messages appear here.
        </Text>
      ) : (
        <Stack gap="2">
          {state.notifications.map((notification) => {
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
                    {notification.kind}
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
