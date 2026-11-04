import type { WorkbenchNotification, WorkbenchNotificationKind } from '@workbench/projectContracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Badge, HStack, Icon, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { queueCommands } from '@features/queue';
import { Button, IconButton } from '@platform/ui/Button';
import { Scrollable } from '@platform/ui/Scrollable';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { createEditorNotificationActionContext } from '@workbench/shell/notifications/editorNotificationActions';
import {
  resolveNotificationActions,
  type NotificationActionContext,
} from '@workbench/shell/notifications/notificationActions';
import { resolveNotificationText } from '@workbench/shell/notifications/notificationToasts';
import { hasNotificationUndo, subscribeNotificationUndos } from '@workbench/shell/notifications/undoRegistry';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useWorkbenchCommands, useWorkbenchQueries, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import {
  BellIcon,
  CircleCheckIcon,
  CircleXIcon,
  InfoIcon,
  TriangleAlertIcon,
  XIcon,
  type LucideIcon,
} from 'lucide-react';
import { memo, useCallback, useMemo, useRef, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

import { NotificationsHeaderActions } from './NotificationsHeaderActions';

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

  // The popover renders the view bare (no widget-frame header), so the header
  // actions that frame would have mounted live here instead.
  if (region === 'popover') {
    return (
      <Stack flex="1" gap="0" minH="0">
        <HStack borderBottomWidth="1px" borderColor="border.subtle" justify="space-between" px="2" py="1.5">
          <Text color="fg.subtle" fontSize="xs" fontWeight="700" textTransform="uppercase">
            {t('widgets.labels.notifications')}
          </Text>
          <NotificationsHeaderActions />
        </HStack>
        <Scrollable minH="0">
          <NotificationsPanel />
        </Scrollable>
      </Stack>
    );
  }

  return <NotificationsPanel />;
};

/** How long a new notice must be on screen before it counts as read. */
const SEEN_AFTER_MS = 1_500;

/** Share of a card that must show for it to count as seen. */
const SEEN_THRESHOLD = 0.5;

const NotificationsPanel = () => {
  const { i18n, t } = useTranslation();
  const notifications = useWorkbenchSelector((snapshot) => snapshot.notifications);
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const focus = useWorkbenchFocus();
  const { getWidgetById, getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const panelRef = useRef<HTMLDivElement>(null);
  // The same buttons the toasts offer, resolved against this editor at render time.
  const actionContext = useMemo(
    () =>
      createEditorNotificationActionContext({
        focus,
        getWidgetById,
        getWidgetsForRegion,
        notifications: commands.notifications,
        queries,
        retryQueueItems: queueCommands.retryItems,
        t: i18n.t,
        widgets: commands.widgets,
      }),
    [commands, focus, getWidgetById, getWidgetsForRegion, i18n, queries]
  );
  /**
   * Focus moves to the next notice's dismiss, else the previous one's. With none left the header's buttons are
   * disabled, so it lands on the list itself, which then shows the empty state and draws a focus ring.
   */
  const dismiss = useCallback(
    (notificationId: string) => {
      const buttons = [...(panelRef.current?.querySelectorAll<HTMLButtonElement>('[data-notification-dismiss]') ?? [])];
      const index = buttons.findIndex((button) => button.dataset.notificationDismiss === notificationId);

      (buttons[index + 1] ?? buttons[index - 1] ?? panelRef.current)?.focus();
      commands.notifications.dismiss(notificationId);
    },
    [commands]
  );
  const markSeen = useCallback(
    (notificationId: string) => commands.notifications.markRead([notificationId]),
    [commands]
  );

  return (
    <Stack
      ref={panelRef}
      aria-label={t('widgets.labels.notifications')}
      flex="1"
      focusVisibleRing="inside"
      gap="3"
      minH="0"
      p="2"
      tabIndex={-1}
    >
      {notifications.length === 0 ? (
        <Text color="fg.subtle" fontSize="xs">
          {t('notifications.empty')}
        </Text>
      ) : (
        <Stack aria-label={t('widgets.labels.notifications')} as="ul" gap="2" listStyle="none">
          {notifications.map((notification) => (
            <NotificationCard
              key={notification.id}
              actionContext={actionContext}
              notification={notification}
              onDismiss={dismiss}
              onSeen={markSeen}
            />
          ))}
        </Stack>
      )}
    </Stack>
  );
};

const NotificationCard = memo(
  ({
    actionContext,
    notification,
    onDismiss,
    onSeen,
  }: {
    actionContext: NotificationActionContext;
    notification: WorkbenchNotification;
    onDismiss: (notificationId: string) => void;
    onSeen: (notificationId: string) => void;
  }) => {
    const { t } = useTranslation();
    /**
     * Seeing a notice is reading it: an unread card that stays at least half on screen for a moment becomes read. One
     * scrolled out of view, or a glance shorter than that (a StrictMode remount included), leaves it new.
     */
    const watchSeen = useCallback(
      (node: HTMLElement | null) => {
        if (!node || notification.isRead) {
          return;
        }

        let timer: ReturnType<typeof setTimeout> | undefined;
        const observer = new IntersectionObserver(
          ([entry]) => {
            if (entry && entry.isIntersecting && entry.intersectionRatio >= SEEN_THRESHOLD) {
              timer ??= setTimeout(() => onSeen(notification.id), SEEN_AFTER_MS);
            } else {
              clearTimeout(timer);
              timer = undefined;
            }
          },
          // Crossing either mark reports; a card sliding under half visible stops its clock.
          { threshold: [0, SEEN_THRESHOLD] }
        );

        observer.observe(node);

        return () => {
          observer.disconnect();
          clearTimeout(timer);
        };
      },
      [notification.id, notification.isRead, onSeen]
    );
    // Undo closures live outside workbench state: re-render when this notice's go (run elsewhere, withdrawn).
    const getOfferedUndoCount = useCallback(
      () =>
        notification.actions?.filter((action) => action.kind === 'undo' && hasNotificationUndo(action.undoId)).length ??
        0,
      [notification.actions]
    );
    useSyncExternalStore(subscribeNotificationUndos, getOfferedUndoCount, getOfferedUndoCount);
    const { message, title } = resolveNotificationText(notification, actionContext.t);
    const actions = resolveNotificationActions(notification.actions ?? [], actionContext);
    const repeats = notification.occurrenceCount ?? 1;
    const IconComponent = kindIcon[notification.kind];
    const dismiss = useCallback(() => onDismiss(notification.id), [notification.id, onDismiss]);

    return (
      <Stack
        ref={watchSeen}
        as="li"
        bg={notification.isRead ? 'bg.subtle' : 'bg.muted'}
        borderWidth="1px"
        borderColor={notification.isRead ? 'border.subtle' : 'border.emphasized'}
        gap="1"
        p="2"
        rounded="md"
      >
        <HStack align="start" gap="2" justify="space-between">
          <HStack align="start" gap="2" minW="0">
            <Icon
              as={IconComponent}
              boxSize="3.5"
              color={`${kindColorPalette[notification.kind]}.300`}
              flexShrink={0}
              mt="0.5"
            />
            <Text fontSize="xs" fontWeight="700" overflowWrap="anywhere">
              <VisuallyHidden>{`${t(`notifications.kind.${notification.kind}`)}: `}</VisuallyHidden>
              {title}
            </Text>
            {repeats > 1 ? (
              <Badge
                colorPalette={kindColorPalette[notification.kind]}
                flexShrink={0}
                fontVariantNumeric="tabular-nums"
              >
                {t('notifications.repeatCount', { count: repeats })}
              </Badge>
            ) : null}
          </HStack>
          <IconButton
            aria-label={t('common.dismissNotification')}
            color="fg.muted"
            data-notification-dismiss={notification.id}
            flexShrink={0}
            size="xs"
            tooltip={t('common.dismissNotification')}
            variant="ghost"
            onClick={dismiss}
          >
            <Icon as={XIcon} boxSize="3" />
          </IconButton>
        </HStack>
        {message ? (
          <Text color="fg.subtle" fontSize="xs" overflowWrap="anywhere">
            {message}
          </Text>
        ) : null}
        <HStack flexWrap="wrap" gap="2" justify="space-between">
          <Text asChild color="fg.subtle" fontSize="xs">
            <time dateTime={notification.createdAt}>{formatRelativeTime(notification.createdAt)}</time>
          </Text>
          {actions.length > 0 ? (
            <HStack flexWrap="wrap" gap="1">
              {actions.map((action) => (
                <Button key={action.label} size="xs" variant="outline" onClick={action.onClick}>
                  {action.label}
                </Button>
              ))}
            </HStack>
          ) : null}
        </HStack>
      </Stack>
    );
  }
);
