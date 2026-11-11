import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, Flex, HStack, Spinner } from '@chakra-ui/react';
import { ListSectionHeader } from '@platform/ui/list/ListSectionHeader';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { BellIcon, TriangleAlertIcon } from 'lucide-react';
import { lazy, Suspense } from 'react';
import { useTranslation } from 'react-i18next';

import { NotificationsHeaderActions } from './NotificationsHeaderActions';

/** The list loads when it is first shown: the editor boots with the status-bar chip alone. */
const NotificationsList = lazy(() =>
  import('./NotificationsList').then((module) => ({ default: module.NotificationsList }))
);

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

  const list = (
    <Suspense fallback={LIST_PENDING}>
      <NotificationsList />
    </Suspense>
  );

  // The popover renders the view bare (no widget-frame header), so the header actions that frame would have mounted
  // live here instead, styled as the list's section header and lined up with its rows.
  if (region === 'popover') {
    return (
      <Box display="flex" flex="1" flexDirection="column" minH="0">
        <HStack
          borderBottomWidth="1px"
          borderColor="border.subtle"
          flexShrink={0}
          justify="space-between"
          pe="2.5"
          ps="1.5"
        >
          <ListSectionHeader count={totalCount} label={t('widgets.labels.notifications')} />
          <NotificationsHeaderActions />
        </HStack>
        {list}
      </Box>
    );
  }

  return list;
};

const NotificationsListPending = () => {
  const { t } = useTranslation();

  return (
    <Flex align="center" aria-label={t('common.loading')} flex="1" justify="center" py="6" role="status">
      <Spinner color="fg.subtle" size="md" />
    </Flex>
  );
};

const LIST_PENDING = <NotificationsListPending />;
