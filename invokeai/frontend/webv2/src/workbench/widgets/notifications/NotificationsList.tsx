/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ListRowProps } from '@platform/ui/list/List';
import type { ListContextMenuAnchor } from '@platform/ui/list/ListItem';
import type { ToastAction } from '@platform/ui/toaster';
import type { WorkbenchNotification, WorkbenchNotificationKind } from '@workbench/projectContracts';

import { Badge, Box, chakra, HStack, Icon, Menu, Portal, VisuallyHidden } from '@chakra-ui/react';
import { queueCommands } from '@features/queue';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { Button, IconButton } from '@platform/ui/Button';
import { EmptyState } from '@platform/ui/EmptyState';
import { List } from '@platform/ui/list/List';
import { ListItem } from '@platform/ui/list/ListItem';
import { listRowsFromItems } from '@platform/ui/list/listRows';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { createEditorNotificationActionContext } from '@workbench/shell/notifications/editorNotificationActions';
import {
  resolveNotificationActions,
  type NotificationActionContext,
} from '@workbench/shell/notifications/notificationActions';
import { resolveNotificationText } from '@workbench/shell/notifications/notificationToasts';
import { hasNotificationUndo, subscribeNotificationUndos } from '@workbench/shell/notifications/undoRegistry';
import { useWorkbenchCommands, useWorkbenchQueries, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { BellIcon, CircleCheckIcon, CircleXIcon, InfoIcon, XIcon, type LucideIcon } from 'lucide-react';
import { memo, useCallback, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

const kindMeta = {
  error: { color: 'fg.error', icon: CircleXIcon },
  info: { color: 'accent.fg', icon: InfoIcon },
  success: { color: 'fg.success', icon: CircleCheckIcon },
} satisfies Record<WorkbenchNotificationKind, { color: string; icon: LucideIcon }>;

/** How long a new notice must be on screen before it counts as read. */
const SEEN_AFTER_MS = 1_500;

/** Share of a row that must show for it to count as seen. */
const SEEN_THRESHOLD = 0.5;

/** A title, a wrapped line of message and the age line; rows are measured, so this only steadies the first scroll. */
const ESTIMATED_ROW_HEIGHT_PX = 64;

const getNotificationKey = (notification: WorkbenchNotification) => notification.id;

interface NotificationMenuTarget extends ListContextMenuAnchor {
  notificationId: string;
}

/**
 * The buttons a notice offers, as its toast offered them. Undo closures live outside workbench state, so the notice's
 * still-offered undos are a snapshot of their own (they go when run elsewhere or withdrawn). The actions are filtered by
 * that snapshot rather than by the registry alone, so memoization sees the change.
 */
const useNotificationActions = (
  notification: WorkbenchNotification,
  actionContext: NotificationActionContext
): ToastAction[] => {
  const getOfferedUndoIds = useCallback(
    () =>
      (notification.actions ?? [])
        .flatMap((action) => (action.kind === 'undo' && hasNotificationUndo(action.undoId) ? [action.undoId] : []))
        .join('\n'),
    [notification.actions]
  );
  const offeredUndoIds = useSyncExternalStore(subscribeNotificationUndos, getOfferedUndoIds, getOfferedUndoIds);
  const offered = (notification.actions ?? []).filter(
    (action) => action.kind !== 'undo' || offeredUndoIds.split('\n').includes(action.undoId)
  );

  return resolveNotificationActions(offered, actionContext);
};

/**
 * The expanded notifications list, loaded with the popover rather than with the status-bar chip. Rows are measured:
 * messages wrap in full, so nothing needs a details view to be read.
 */
export const NotificationsList = () => {
  const { i18n, t } = useTranslation();
  const notifications = useWorkbenchSelector((snapshot) => snapshot.notifications);
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const focus = useWorkbenchFocus();
  const { getWidgetById, getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const panelRef = useRef<HTMLDivElement>(null);
  const rows = useMemo(() => listRowsFromItems(notifications, getNotificationKey), [notifications]);
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
  const [menuTarget, setMenuTarget] = useState<NotificationMenuTarget | null>(null);
  /**
   * Focus moves to the next notice's dismiss, else the previous one's. The List keeps the focused row and its overscan
   * mounted, so a neighbour's dismiss is always in the DOM. With none left the header's buttons are disabled, so focus
   * holds on the panel until the empty state that replaces the list can take it.
   */
  const dismiss = useCallback(
    (notificationId: string) => {
      const panel = panelRef.current;
      const buttons = [...(panel?.querySelectorAll<HTMLButtonElement>('[data-notification-dismiss]') ?? [])];
      const index = buttons.findIndex((button) => button.dataset.notificationDismiss === notificationId);
      const neighbourId = (buttons[index + 1] ?? buttons[index - 1])?.dataset.notificationDismiss;
      const target = () =>
        panel?.querySelector<HTMLElement>(
          neighbourId ? `[data-notification-dismiss="${CSS.escape(neighbourId)}"]` : '[data-notifications-empty]'
        ) ?? null;

      (target() ?? panel)?.focus();
      commands.notifications.dismiss(notificationId);
      // Settle once the list has: the empty state appears only after the commit, and a dismiss chosen from the row's
      // menu (an update outside React's event handling) can remount the neighbour's row, dropping focus to the page.
      requestAnimationFrame(() => {
        const active = document.activeElement;

        if (active === null || active === document.body || active === panel) {
          target()?.focus();
        }
      });
    },
    [commands]
  );
  const openMenu = useCallback(
    (notificationId: string, anchor: ListContextMenuAnchor) => setMenuTarget({ ...anchor, notificationId }),
    []
  );
  const closeMenu = useCallback(() => setMenuTarget(null), []);
  const markSeen = useCallback(
    (notificationId: string) => commands.notifications.markRead([notificationId]),
    [commands]
  );
  const renderItem = useCallback(
    (notification: WorkbenchNotification, rowProps: ListRowProps) => (
      // Spread so the memoized row compares the List's primitives, not a fresh object per render.
      <NotificationRow
        {...rowProps}
        actionContext={actionContext}
        isMenuOpen={menuTarget?.notificationId === notification.id}
        notification={notification}
        onDismiss={dismiss}
        onOpenMenu={openMenu}
        onSeen={markSeen}
      />
    ),
    [actionContext, dismiss, markSeen, menuTarget, openMenu]
  );

  return (
    <Box
      ref={panelRef}
      // A grid area is a definite height for the List to fill, even when the popover sizes to its content up to a cap;
      // a flex column would let the List grow past the cap and clip instead of scroll.
      display="grid"
      flex="1"
      focusVisibleRing="inside"
      gridTemplateRows="minmax(0, 1fr)"
      minH="0"
      px="1.5"
      py="1.5"
      // Holds focus for the moment between dismissing the last notice and the empty state appearing.
      tabIndex={-1}
    >
      <List
        dividers
        emptyState={
          // Named so that focus landing here after the last dismiss says what happened.
          <EmptyState
            aria-label={t('notifications.emptyTitle')}
            data-notifications-empty=""
            description={t('notifications.empty')}
            focusVisibleRing="inside"
            icon={<BellIcon />}
            role="group"
            tabIndex={-1}
            title={t('notifications.emptyTitle')}
          />
        }
        estimatedRowHeight={ESTIMATED_ROW_HEIGHT_PX}
        // The panel pads the rows; the header above lines up with them.
        inset="none"
        label={t('widgets.labels.notifications')}
        renderItem={renderItem}
        rowHeight="measured"
        rows={rows}
      />
      <NotificationMenu
        actionContext={actionContext}
        notifications={notifications}
        target={menuTarget}
        onClose={closeMenu}
        onDismiss={dismiss}
      />
    </Box>
  );
};

/**
 * One menu for the list, opened from a row's button or by right-click: the row's actions and its dismiss. An item that
 * moved focus on (a dismiss to the next notice, Show in queue to the queue) keeps it; otherwise focus returns to the
 * row.
 */
const NotificationMenu = ({
  actionContext,
  notifications,
  onClose,
  onDismiss,
  target,
}: {
  actionContext: NotificationActionContext;
  notifications: readonly WorkbenchNotification[];
  onClose: () => void;
  onDismiss: (notificationId: string) => void;
  target: NotificationMenuTarget | null;
}) => {
  // Kept through the exit animation; keyed by notice so another row's menu still opens at its own anchor.
  const { release, value: shown } = useExitRetainedValue(target);
  const notification = shown
    ? (notifications.find((candidate) => candidate.id === shown.notificationId) ?? null)
    : null;

  return (
    <Menu.Root
      key={shown ? shown.notificationId : 'none'}
      lazyMount
      open={target !== null}
      positioning={{
        getAnchorRect: () => (shown ? { height: 1, width: 1, x: shown.x, y: shown.y } : null),
        placement: 'bottom-start',
      }}
      unmountOnExit
      onExitComplete={release}
      onOpenChange={(event) => {
        if (event.open) {
          return;
        }

        const active = document.activeElement;

        if (!active || active === document.body || active.closest('[data-scope="menu"]')) {
          target?.restoreFocus();
        }
        onClose();
      }}
    >
      <Portal>
        <Menu.Positioner>
          {notification ? (
            <NotificationMenuItems actionContext={actionContext} notification={notification} onDismiss={onDismiss} />
          ) : null}
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const NotificationMenuItems = ({
  actionContext,
  notification,
  onDismiss,
}: {
  actionContext: NotificationActionContext;
  notification: WorkbenchNotification;
  onDismiss: (notificationId: string) => void;
}) => {
  const { t } = useTranslation();
  const actions = useNotificationActions(notification, actionContext);

  return (
    <MenuContent minW="12rem">
      {actions.map((action) => (
        <MenuActionItem key={action.label} label={action.label} value={action.label} onSelect={action.onClick} />
      ))}
      {actions.length > 0 ? <Menu.Separator /> : null}
      <MenuActionItem
        icon={XIcon}
        label={t('common.dismissNotification')}
        value="dismiss"
        onSelect={() => onDismiss(notification.id)}
      />
    </MenuContent>
  );
};

const NotificationRow = memo(
  ({
    actionContext,
    density,
    isMenuOpen,
    itemKey,
    notification,
    onDismiss,
    onOpenMenu,
    onSeen,
    positionInSet,
    setSize,
    tabIndex,
  }: Pick<ListRowProps, 'density' | 'itemKey' | 'positionInSet' | 'setSize' | 'tabIndex'> & {
    actionContext: NotificationActionContext;
    isMenuOpen: boolean;
    notification: WorkbenchNotification;
    onDismiss: (notificationId: string) => void;
    onOpenMenu: (notificationId: string, anchor: ListContextMenuAnchor) => void;
    onSeen: (notificationId: string) => void;
  }) => {
    const { t } = useTranslation();
    /**
     * Seeing a notice is reading it: an unread row that stays at least half on screen for a moment becomes read. One
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
          // Crossing either mark reports; a row sliding under half visible stops its clock.
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
    const actions = useNotificationActions(notification, actionContext);
    const { message, title } = resolveNotificationText(notification, actionContext.t);
    const repeats = notification.occurrenceCount ?? 1;
    const kind = kindMeta[notification.kind];
    const dismiss = useCallback(() => onDismiss(notification.id), [notification.id, onDismiss]);
    // The row's button opens its menu (keyboard, assistive tech, touch); a mouse right-clicks.
    const openMenu = useCallback(
      (anchor: ListContextMenuAnchor) => onOpenMenu(notification.id, anchor),
      [notification.id, onOpenMenu]
    );

    // The wrapper is the list item so the seen observer can watch the whole row; the List places and measures it.
    return (
      <Box ref={watchSeen} aria-posinset={positionInSet} aria-setsize={setSize} role="listitem">
        <ListItem
          actions={
            // Level with the title rather than centred on a message that wraps over several lines.
            <Box alignSelf="flex-start" pt="1">
              <IconButton
                aria-label={t('common.dismissNotification')}
                color="fg.muted"
                data-notification-dismiss={notification.id}
                size="xs"
                tooltip={t('common.dismissNotification')}
                variant="ghost"
                onClick={dismiss}
              >
                <Icon as={XIcon} boxSize="3.5" />
              </IconButton>
            </Box>
          }
          badges={
            repeats > 1 ? (
              <Badge fontSize="xs" fontVariantNumeric="tabular-nums" variant="surface">
                {t('notifications.repeatCount', { count: repeats })}
              </Badge>
            ) : undefined
          }
          density={density}
          description={
            // Phrasing content only: the text sits inside the row's button.
            <chakra.span display="flex" flexDirection="column" gap="0.5">
              {message ? <chakra.span overflowWrap="anywhere">{message}</chakra.span> : null}
              <chakra.time color="fg.subtle" dateTime={notification.createdAt}>
                {formatRelativeTime(notification.createdAt)}
              </chakra.time>
            </chakra.span>
          }
          detail={
            actions.length > 0 ? (
              // Inset past the kind tile, under the text the actions belong to.
              <HStack flexWrap="wrap" gap="1" ps="10">
                {actions.map((action) => (
                  <Button key={action.label} size="xs" variant="outline" onClick={action.onClick}>
                    {action.label}
                  </Button>
                ))}
              </HStack>
            ) : undefined
          }
          isMenuOpen={isMenuOpen}
          itemKey={itemKey}
          leading={
            <>
              <KindTile color={kind.color} icon={kind.icon} isUnread={!notification.isRead} />
              {/* Kind and unread state lead the row's name; the tile's icon shape and dot are their visual cues. */}
              <VisuallyHidden>
                {notification.isRead ? null : `${t('notifications.unread')}, `}
                {`${t(`notifications.kind.${notification.kind}`)}: `}
              </VisuallyHidden>
            </>
          }
          role="presentation"
          tabIndex={tabIndex}
          title={title}
          titleTruncate="end"
          onContextMenu={openMenu}
        />
      </Box>
    );
  }
);

/** The kind's icon on a tile, as the queue's rows show their state; an unread notice adds a dot to its corner. */
const KindTile = ({ color, icon, isUnread }: { color: string; icon: LucideIcon; isUnread: boolean }) => (
  <chakra.span
    alignItems="center"
    alignSelf="flex-start"
    aria-hidden
    bg="bg.subtle"
    boxSize="8"
    color={color}
    display="flex"
    flexShrink={0}
    justifyContent="center"
    position="relative"
    rounded="md"
  >
    <Icon as={icon} boxSize="4" />
    {isUnread ? (
      <chakra.span
        bg="accent.solid"
        borderColor="bg.panel"
        borderWidth="2px"
        boxSize="2.5"
        insetEnd="-0.5"
        position="absolute"
        rounded="full"
        top="-0.5"
      />
    ) : null}
  </chakra.span>
);
