/* eslint-disable react-perf/jsx-no-jsx-as-prop */
import type { QueueItemReadModel } from '@features/queue/core/types';

import { Box, HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { getResultImageName } from '@features/queue/core/generationMeta';
import { getQueueItemReadModelLabel } from '@features/queue/core/itemLabels';
import { parseQueueItemOriginProjectId } from '@features/queue/data/events';
import { useItemProgress } from '@features/queue/data/itemProgressStore';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { ListItem, type ListContextMenuAnchor } from '@platform/ui/list/ListItem';
import { MenuContent, MenuPartPlaceholder, useContextMenu, useMenuPart } from '@platform/ui/Menu';
import { ChevronRightIcon } from 'lucide-react';
import { memo, useCallback, useMemo, useRef, useState, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { CancelQueueItemButton } from './CancelQueueItemButton';
import { formatCompactAge, formatDuration } from './formatDuration';
import { QueueItemDetails } from './QueueItemDetails';
import { QueueItemThumbnail } from './QueueItemThumbnail';
import { isCancellableQueueItem, isRetryableQueueItem } from './queueOwnership';
import { QueueStatusDot } from './QueueStatusDot';
import { QueueStepProgress } from './QueueStepProgress';
import { useLoadingModelLabel, useQueueUi } from './QueueUiContext';
import { clearPendingQueueItemReveal, type QueueItemRevealRequest } from './queueUiStore';
import { RetryQueueItemButton } from './RetryQueueItemButton';
import { getStatusMeta } from './statusMeta';
import { useDeviceLabel } from './useDeviceLabel';

const CHEVRON_OPEN = { transform: 'rotate(90deg)' } as const;

export const QueueItemRow = memo(
  ({ item, revealRequest }: { item: QueueItemReadModel; revealRequest?: QueueItemRevealRequest | null }) => {
    const { t } = useTranslation();
    const { canManageItem, canViewItemDetails, isProjectOpen, itemRecallMenuGroup, preloadItemActions } = useQueueUi();
    const [expanded, setExpanded] = useState(false);
    // Safari and Firefox on macOS do not focus a clicked button; the row's A key acts on the focused row. A click must
    // not show the keyboard ring: `focusVisible: false` says so where supported (Firefox), and elsewhere script focus
    // matches :focus-visible only when the element focus left from did.
    const toggle = useCallback((event: MouseEvent<HTMLButtonElement>) => {
      if (document.activeElement !== event.currentTarget) {
        event.currentTarget.focus({ focusVisible: false, preventScroll: true });
      }
      setExpanded((open) => !open);
    }, []);
    const menu = useContextMenu();
    // Kept through the exit animation: the anchor clears the moment the menu closes.
    const { release: releaseMenu, value: shownMenuAnchor } = useExitRetainedValue(menu.anchor);
    // Warm the actions and the menu's recall entries on intent, so the first expand or menu has nothing to wait for.
    const preload = useCallback(() => {
      preloadItemActions?.();
      itemRecallMenuGroup.load();
    }, [itemRecallMenuGroup, preloadItemActions]);
    // The keyboard's menu key and assistive technology open the menu without intent first.
    const { open: openContextMenu } = menu;
    const openMenu = useCallback(
      (anchor: ListContextMenuAnchor) => {
        itemRecallMenuGroup.load();
        openContextMenu(anchor);
      },
      [itemRecallMenuGroup, openContextMenu]
    );
    const consumedRevealRequestIdRef = useRef<number | null>(null);
    const duration = formatDuration(item.startedAt, item.completedAt);
    const age = formatCompactAge(item.completedAt ?? item.createdAt);
    const ageLabel = [duration, age].filter(Boolean).join(' · ');
    const isFailed = item.status === 'failed';
    const canManage = canManageItem(item);
    const isCancellable = isCancellableQueueItem(item) && canManage;
    const isRetryable = isRetryableQueueItem(item) && canManage;
    const originProjectId = parseQueueItemOriginProjectId(item.origin);
    // A Canvas rerun is staged by a tab with its project open, which need not be this one (or this user's).
    const isRetryForProjectClosedHere =
      item.destination === 'canvas' && originProjectId !== null && !isProjectOpen(originProjectId);
    const canExpand = canViewItemDetails(item);

    const consumeReveal = useCallback(
      (node: HTMLDivElement | null) => {
        if (!node || !revealRequest || consumedRevealRequestIdRef.current === revealRequest.requestId) {
          return;
        }

        consumedRevealRequestIdRef.current = revealRequest.requestId;
        if (canExpand) {
          setExpanded(true);
        }
        node.scrollIntoView({ block: 'nearest' });
        clearPendingQueueItemReveal(revealRequest.requestId);
      },
      [canExpand, revealRequest]
    );

    const progress = useItemProgress(item.id);
    const liveImage = progress?.image ?? null;
    const resultImageName = getResultImageName(item);
    const statusLabel = t(getStatusMeta(item.status).labelKey);
    // A running item's device arrives on the progress event before the row's DTO is
    // refetched, so prefer the live value and fall back to the persisted one.
    const deviceLabel = useDeviceLabel(progress?.device ?? item.device);

    // Rows read by what they made; the id stays alongside for matching against logs and other tabs.
    const itemLabel = useMemo(() => getQueueItemReadModelLabel(item), [item]);
    const itemNumber = t('widgets.queue.itemNumber', { id: item.id });
    const errorLine = isFailed && !expanded ? item.errorMessage?.trim().split('\n')[0] : undefined;

    const showBorder = expanded || isFailed;
    const borderColor = showBorder ? (isFailed ? 'fg.error' : 'border') : 'transparent';

    // The card is the list item: progress and details belong to the row, not to the list.
    return (
      <Box
        ref={consumeReveal}
        borderColor={borderColor}
        borderWidth={1}
        // The `queue.recallAll` hotkey claims its key only on rows that offer recall.
        data-queue-recall-item={canExpand ? item.id : undefined}
        overflow="hidden"
        role="listitem"
        rounded="md"
      >
        <ListItem
          actions={
            isCancellable ? (
              <CancelQueueItemButton itemId={item.id} />
            ) : isRetryable ? (
              <RetryQueueItemButton isProjectClosedHere={isRetryForProjectClosedHere} itemId={item.id} />
            ) : undefined
          }
          description={
            <HStack gap="1.5" minW="0">
              <QueueStatusDot status={item.status} />
              <Text
                fontVariantNumeric="tabular-nums"
                title={deviceLabel ? t('widgets.queue.device.tooltip', { name: deviceLabel.name }) : undefined}
                truncate
              >
                {[
                  itemLabel ? itemNumber : null,
                  statusLabel,
                  ageLabel,
                  deviceLabel ? t('widgets.queue.device.shortLabel', { index: deviceLabel.index }) : null,
                ]
                  .filter(Boolean)
                  .join(' · ')}
              </Text>
            </HStack>
          }
          isExpanded={canExpand ? expanded : undefined}
          isMenuOpen={menu.anchor !== null}
          leading={<QueueItemThumbnail boxSize="8" imageName={resultImageName} liveImage={liveImage} />}
          role="presentation"
          title={itemLabel || t('widgets.queue.itemTitle', { id: item.id })}
          trailing={
            canExpand ? (
              <Icon
                as={ChevronRightIcon}
                boxSize="4"
                transition="transform var(--wb-motion-duration-fast) ease"
                css={expanded ? CHEVRON_OPEN : undefined}
              />
            ) : undefined
          }
          onContextMenu={canExpand ? openMenu : undefined}
          onIntent={canExpand ? preload : undefined}
          onPress={canExpand ? toggle : undefined}
        />
        {shownMenuAnchor ? (
          <QueueItemContextMenu
            anchor={shownMenuAnchor}
            item={item}
            open={menu.anchor !== null}
            onClose={menu.close}
            onExitComplete={releaseMenu}
            onRequestDismiss={menu.onRequestDismiss}
          />
        ) : null}

        {item.status === 'in_progress' ? (
          <Box px="2.5" pb="2">
            <RunningItemProgress message={progress?.message ?? ''} percentage={progress?.percentage ?? null} />
          </Box>
        ) : null}

        {errorLine ? (
          <Text color="fg.error" fontSize="xs" pb="2" px="2.5" truncate>
            {errorLine}
          </Text>
        ) : null}

        {expanded ? (
          <Box pb="2.5" pt="1" px="2.5">
            <QueueItemDetails item={item} />
          </Box>
        ) : null}
      </Box>
    );
  }
);

/** A running row's progress; with no step yet while the server loads a model, it says which instead of only sweeping. */
const RunningItemProgress = ({ message, percentage }: { message: string; percentage: number | null }) => {
  const { t } = useTranslation();
  const loadingModel = useLoadingModelLabel();
  const caption = message.trim() || (loadingModel ? t('widgets.queue.loadingModel', { model: loadingModel }) : '');

  return <QueueStepProgress message={caption} percentage={percentage} />;
};

/** Mounted only while open or closing, so a long queue keeps one menu machine, and one recall read, at most. */
const QueueItemContextMenu = ({
  anchor,
  item,
  onClose,
  onExitComplete,
  onRequestDismiss,
  open,
}: {
  anchor: { x: number; y: number };
  item: QueueItemReadModel;
  onClose: () => void;
  onExitComplete: () => void;
  onRequestDismiss: (event: Event) => void;
  open: boolean;
}) => {
  const { itemRecallMenuGroup } = useQueueUi();
  const recall = useMenuPart(itemRecallMenuGroup);
  const positioning = useMemo(
    () => ({
      getAnchorRect: () => ({ height: 1, width: 1, x: anchor.x, y: anchor.y }),
      placement: 'bottom-start' as const,
    }),
    [anchor.x, anchor.y]
  );
  const onOpenChange = useCallback(
    (details: { open: boolean }) => {
      if (!details.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Menu.Root
      lazyMount
      open={open}
      positioning={positioning}
      unmountOnExit
      onExitComplete={onExitComplete}
      onOpenChange={onOpenChange}
      onRequestDismiss={onRequestDismiss}
    >
      <Portal>
        <Menu.Positioner>
          <MenuContent aria-busy={recall.status === 'loading'} minW="13rem" py="1">
            {/* The menu opens at once, so Escape and an outside press work while its entries load. */}
            {recall.status === 'loaded' ? <recall.part item={item} /> : <MenuPartPlaceholder status={recall.status} />}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
