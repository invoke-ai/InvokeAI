import type { WidgetInstanceId } from '@workbench/widgetContracts';
import type { WidgetRegionDropState } from '@workbench/widgetDnd';
import type { PlacedWidgetRegionItem, WidgetPlacementInstanceMeta } from '@workbench/widgetRegionViewModel';

import { Box, Flex, Icon, Popover, Portal } from '@chakra-ui/react';
import { useDndContext, useDroppable } from '@dnd-kit/core';
import { horizontalListSortingStrategy } from '@dnd-kit/sortable';
import { PopoverContent, Row, Tooltip } from '@platform/ui';
import { Button } from '@platform/ui/Button';
import { isModalPresent, subscribeModalPresence } from '@platform/ui/modalPresence';
import { useHighlightedRegion } from '@workbench/focusRegions';
import { useShortcutFocusRestore } from '@workbench/hotkeys/hintSources';
import { isShortcutGuideSessionKey } from '@workbench/hotkeys/shortcutHints';
import {
  WidgetEnableMenu,
  WidgetInstanceContextMenu,
  WidgetRegionDropOverlay,
  WidgetRendererById,
  WidgetStrip,
  useWidgetSortable,
  type WidgetEnableMenuItem,
  type WidgetInstanceContextMenuTarget,
} from '@workbench/widget-frame';
import {
  CompactWidgetCapacityProvider,
  createCompactWidgetCapacity,
  useCompactWidgetCapacityResource,
} from '@workbench/widget-frame/compactWidgetCapacity';
import { CompactWidgetFailure } from '@workbench/widget-frame/WidgetFailureBoundary';
import { useWidgetRecovery } from '@workbench/widget-frame/widgetRecovery';
import { WidgetRecoveryAnnouncement } from '@workbench/widget-frame/WidgetRenderer';
import {
  getWidgetRegionDropId,
  getWidgetRegionEndDropData,
  getWidgetRegionEndDropId,
  isWidgetInstanceDragData,
} from '@workbench/widgetDnd';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { closeWidgetPlacement, openWidgetPlacement, revealWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import {
  createWidgetRegionViewModelFromState,
  getWidgetRegionItems,
  isCompactBottomItem,
  isExpandableBottomItem,
  isPopoverBottomItem,
} from '@workbench/widgetRegionViewModel';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { ArrowRightToLineIcon } from 'lucide-react';
import {
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
  useCallback,
  useMemo,
  useState,
  useRef,
  useSyncExternalStore,
} from 'react';
import { useTranslation } from 'react-i18next';

interface BottomWidgetItem extends PlacedWidgetRegionItem<WidgetPlacementInstanceMeta> {
  isExpandable: boolean;
  isPopover: boolean;
}

const BOTTOM_MENU_POSITIONING = { placement: 'top-end' } as const;
const WIDGET_POPOVER_POSITIONING = { placement: 'top-end' } as const;
const BOTTOM_MENU_TRIGGER = { kind: 'bottom' } as const;
/** Use content color for active brand accents; background tint is indistinguishable on the light theme. */
const COMPACT_ROW_HOVER_PROPS = { bg: 'bg.hover', color: 'fg' };

/** Size drop overlays per cluster so one strip-wide curtain cannot cover the other target. */
const ClusterDropRing = ({ dropState, isOver }: { dropState: WidgetRegionDropState; isOver: boolean }) => (
  <WidgetRegionDropOverlay dropState={dropState} isOver={isOver} left="-4px" right="-4px" zIndex={3} />
);

/** The trailing cluster and its "move to the right side" drop target. */
const BottomEndCluster = ({
  children,
  dropState,
  showDropChrome,
}: {
  children: ReactNode;
  dropState: WidgetRegionDropState;
  showDropChrome: boolean;
}) => {
  const { isOver, setNodeRef } = useDroppable({
    data: getWidgetRegionEndDropData('bottom'),
    id: getWidgetRegionEndDropId('bottom'),
  });

  return (
    <Flex
      ref={setNodeRef}
      data-status-bar-cluster
      align="center"
      alignSelf="stretch"
      flexShrink={0}
      position="relative"
    >
      {children}
      {showDropChrome ? (
        <>
          <Flex align="center" justify="center" minW="10">
            <Icon as={ArrowRightToLineIcon} boxSize="3" color={isOver ? 'fg' : 'fg.muted'} zIndex={4} />
          </Flex>
          <ClusterDropRing dropState={dropState} isOver={isOver} />
        </>
      ) : null}
    </Flex>
  );
};
const COMPACT_ROW_ACTIVE_PROPS = { bg: 'bg.emphasized', color: 'brand.fg' };
const COMPACT_ROW_ACTIVE_HOVER_PROPS = { bg: 'bg.emphasized', color: 'brand.fg' };
const TOOLTIP_POSITIONING = { placement: 'top' } as const;

export const StatusBar = ({ dropState }: { dropState: WidgetRegionDropState }) => {
  const [capacity] = useState(createCompactWidgetCapacity);
  const bindSpacer = useCallback((element: HTMLDivElement | null) => capacity.bindSpacer(element), [capacity]);
  const { t } = useTranslation();
  const placementProject = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const bottomRegion = useActiveProjectSelector((project) => project.widgetRegions.bottom);
  const { widgets } = useWorkbenchCommands();
  const [enableMenuTarget, setEnableMenuTarget] = useState<{ x: number; y: number } | null>(null);
  const [instanceMenuTarget, setInstanceMenuTarget] = useState<WidgetInstanceContextMenuTarget | null>(null);
  const getWidgetLabel = useCallback(
    (manifest: Parameters<typeof resolveWidgetLabel>[0]) => resolveWidgetLabel(manifest, t),
    [t]
  );
  const bottomRegionViewModel = createWidgetRegionViewModelFromState({
    getWidgetLabel,
    region: 'bottom',
    regionState: bottomRegion,
    widgetInstances: placementProject.widgetInstances,
    widgets: getWidgetsForRegion('bottom'),
  });
  const items = getWidgetRegionItems(bottomRegionViewModel);
  const compactItems = items.flatMap((item): BottomWidgetItem[] => {
    if (!isCompactBottomItem(item)) {
      return [];
    }

    return [{ ...item, isExpandable: isExpandableBottomItem(item), isPopover: isPopoverBottomItem(item) }];
  });
  // Preserve instance order within clusters; only rendering splits them. Context menus, spacer drops, or
  // neighboring widgets move membership.
  const alignEndIds = useMemo(
    () => new Set(bottomRegion.alignEndInstanceIds ?? []),
    [bottomRegion.alignEndInstanceIds]
  );
  const startItems = compactItems.filter((item) => !alignEndIds.has(item.id));
  const endItems = compactItems.filter((item) => alignEndIds.has(item.id));
  const sortableInstanceIds = useMemo(
    () => [...startItems.map((item) => item.id), ...endItems.map((item) => item.id)],
    [startItems, endItems]
  );
  const openEnableMenu = useCallback((event: MouseEvent) => {
    event.preventDefault();
    setEnableMenuTarget({ x: event.clientX, y: event.clientY });
  }, []);
  const openInstanceMenu = useCallback((item: BottomWidgetItem, event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setInstanceMenuTarget({ item, x: event.clientX, y: event.clientY });
  }, []);
  const toggleBottomWidget = useCallback(
    (item: WidgetEnableMenuItem) =>
      item.isEnabled
        ? closeWidgetPlacement({
            widgets,
            getWidgetById,
            instanceId: item.id,
            project: placementProject,
            region: 'bottom',
          })
        : openWidgetPlacement({
            widgets,
            getWidgetsForRegion,
            options: { createNew: item.allowMultiple, preferredRegions: ['bottom'] },
            typeId: item.typeId,
          }),
    [placementProject, widgets]
  );
  const handleSelect = useCallback(
    (instanceId: WidgetInstanceId) =>
      revealWidgetPlacement({ instanceId, project: placementProject, region: 'bottom', widgets }),
    [placementProject, widgets]
  );
  const setItemAlignment = useCallback(
    (item: WidgetEnableMenuItem, align: 'start' | 'end') =>
      widgets.setAlignment({ align, instanceId: item.id, region: 'bottom' }),
    [widgets]
  );
  const isItemAlignedEnd = useCallback((item: WidgetEnableMenuItem) => alignEndIds.has(item.id), [alignEndIds]);
  const dnd = useDndContext();
  const showDropChrome =
    dropState.isActive && dropState.isAllowed && isWidgetInstanceDragData(dnd.active?.data.current);
  // Accent only zone-background hover; chip hover previews reordering.
  const isOverStart = showDropChrome && String(dnd.over?.id ?? '') === getWidgetRegionDropId('bottom');
  const handleContextClose = useCallback(() => setEnableMenuTarget(null), []);
  const handleInstanceClose = useCallback(() => setInstanceMenuTarget(null), []);
  // The bottom panel's outline is drawn over this border.
  const isBottomOutlined = useHighlightedRegion() === 'bottom';

  return (
    <CompactWidgetCapacityProvider capacity={capacity}>
      <WidgetStrip
        align="center"
        as="footer"
        bg="bg.subtle"
        borderTopWidth="1px"
        borderColor={isBottomOutlined ? 'transparent' : 'border.subtle'}
        color="fg.muted"
        dropState={dropState}
        flexShrink={0}
        h="6"
        overlay="none"
        px="2"
        region="bottom"
        sortableInstanceIds={sortableInstanceIds}
        strategy={horizontalListSortingStrategy}
        w="full"
        onContextMenu={openEnableMenu}
      >
        <Flex data-status-bar-cluster align="center" alignSelf="stretch" flexShrink={0} position="relative">
          {startItems.map((item) => (
            <CompactBottomWidget
              key={item.id}
              item={item}
              isActive={item.isExpandable && item.id === bottomRegion.activeInstanceId && !bottomRegion.isCollapsed}
              onContextMenu={openInstanceMenu}
              onSelect={handleSelect}
            />
          ))}

          <WidgetEnableMenu
            contextTarget={enableMenuTarget}
            groupLabel="Bottom Widgets"
            items={items}
            positioning={BOTTOM_MENU_POSITIONING}
            trigger={BOTTOM_MENU_TRIGGER}
            triggerLabel="Bottom widget visibility"
            onContextClose={handleContextClose}
            onToggle={toggleBottomWidget}
          />
          {showDropChrome ? <ClusterDropRing dropState={dropState} isOver={isOverStart} /> : null}
        </Flex>

        <Box ref={bindSpacer} flex="1" minW="0" />
        <BottomEndCluster dropState={dropState} showDropChrome={showDropChrome}>
          {endItems.map((item) => (
            <CompactBottomWidget
              key={item.id}
              item={item}
              isActive={item.isExpandable && item.id === bottomRegion.activeInstanceId && !bottomRegion.isCollapsed}
              onContextMenu={openInstanceMenu}
              onSelect={handleSelect}
            />
          ))}
        </BottomEndCluster>
        <WidgetInstanceContextMenu
          isAlignedEnd={isItemAlignedEnd}
          target={instanceMenuTarget}
          onClose={handleInstanceClose}
          onRemove={toggleBottomWidget}
          onSetAlignment={setItemAlignment}
        />
      </WidgetStrip>
    </CompactWidgetCapacityProvider>
  );
};

const CompactBottomWidget = ({
  isActive,
  item,
  onContextMenu,
  onSelect,
}: {
  isActive: boolean;
  item: BottomWidgetItem;
  onContextMenu: (item: BottomWidgetItem, event: MouseEvent) => void;
  onSelect: (widgetId: WidgetInstanceId) => void;
}) => {
  const recovery = useWidgetRecovery(item.id, item.widget);
  const modal = useSyncExternalStore(subscribeModalPresence, isModalPresent, isModalPresent);
  const suppressed = modal && item.widget.manifest.preserveWorkbenchFocus && !recovery.hasFailure;
  const { dragHandleProps, isDragging, setNodeRef, style } = useWidgetSortable({
    disabled: suppressed,
    instanceId: item.id,
    region: 'bottom',
    typeId: item.typeId,
  });
  const { t } = useTranslation();
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);
  const restoreShortcutFocus = useShortcutFocusRestore();
  const capacity = useCompactWidgetCapacityResource();
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const statusFocusTarget = useCallback(
    () =>
      wrapperRef.current?.querySelector<HTMLElement>(
        '[data-scope="popover"][data-part="trigger"]:not([disabled]):not([data-disabled]):not([aria-disabled="true"])'
      ) ?? wrapperRef.current,
    []
  );
  const bindWrapper = useCallback(
    (element: HTMLDivElement | null) => {
      wrapperRef.current = element;
      setNodeRef(element);
      const release =
        item.widget.manifest.compactSizing === 'remaining-space' ? capacity?.bindChip(element) : undefined;
      return () => {
        wrapperRef.current = null;
        release?.();
        setNodeRef(null);
      };
    },
    [capacity, item.widget.manifest.compactSizing, setNodeRef]
  );
  const tooltipError = recovery.getError('tooltip');
  const isActivatable = !recovery.error && (item.isExpandable || item.isPopover);
  const rowDragHandleProps = recovery.error
    ? {}
    : isActivatable
      ? Object.fromEntries(Object.entries(dragHandleProps).filter(([key]) => key !== 'onKeyDown'))
      : dragHandleProps;
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLElement>) => {
      if (item.widget.manifest.preserveWorkbenchFocus) {
        if (!isShortcutGuideSessionKey(event)) {
          return;
        }
        event.stopPropagation();
      }
      if (event.key !== 'Enter' && event.key !== ' ') {
        return;
      }

      event.preventDefault();

      // A popover chip is a real Popover.Trigger; a div gets no native
      // Enter/Space click, so synthesize one for the trigger's own handling.
      if (item.isPopover) {
        event.currentTarget.click();
        return;
      }

      onSelect(item.id);
    },
    [item.id, item.isPopover, item.widget.manifest.preserveWorkbenchFocus, onSelect]
  );
  const activationProps = useMemo(
    () =>
      isActivatable
        ? {
            role: 'button' as const,
            tabIndex: 0,
            onKeyDown: handleKeyDown,
          }
        : {},
    [handleKeyDown, isActivatable]
  );
  const handleClick = useCallback(
    (event: MouseEvent) => {
      if (recovery.error) {
        event.preventDefault();
        return;
      }
      if (item.isExpandable && !recovery.error) {
        onSelect(item.id);
      }
    },
    [item.id, item.isExpandable, onSelect, recovery.error]
  );
  const openFailedPanel = useCallback(() => {
    if (item.isPopover) {
      setIsPopoverOpen(true);
    } else if (item.isExpandable) {
      onSelect(item.id);
    }
  }, [item.id, item.isExpandable, item.isPopover, onSelect]);
  const failureOpenButton = useMemo(
    () => (
      <Button
        aria-label={`${t('widgets.failure.title', { label: item.label })} — ${t('common.open')}`}
        size="xs"
        variant="ghost"
        onClick={openFailedPanel}
      >
        {t('common.open')}
      </Button>
    ),
    [item.label, openFailedPanel, t]
  );
  const handleContextMenu = useCallback((event: MouseEvent) => onContextMenu(item, event), [item, onContextMenu]);
  const handlePopoverOpenChange = useCallback((event: { open: boolean }) => setIsPopoverOpen(event.open), []);
  const tooltipContent = useMemo(() => (item.instance ? <BottomWidgetTooltipContent item={item} /> : null), [item]);
  const isRowActive = item.isPopover ? isPopoverOpen : isActive;

  // Keep failed recovery mounted through its own modal; healthy guides close and release their measured DOM.
  if (suppressed) {
    if (isPopoverOpen) {
      setIsPopoverOpen(false);
    }
    return null;
  }

  // Named by its compact content, which shows (or visually hides) the widget's name: an aria-label would
  // replace the visible status text that speech-input users read off the chip (WCAG 2.5.3).
  const row = (
    <Row
      {...rowDragHandleProps}
      aria-pressed={item.isPopover || recovery.error ? undefined : isRowActive}
      color={isRowActive ? undefined : 'fg.muted'}
      cursor={isDragging ? 'grabbing' : 'default'}
      px={item.widget.manifest.compactSizing === 'remaining-space' ? '1.5' : undefined}
      h="full"
      w="auto"
      {...(isRowActive ? COMPACT_ROW_ACTIVE_PROPS : null)}
      _hover={isRowActive ? COMPACT_ROW_ACTIVE_HOVER_PROPS : COMPACT_ROW_HOVER_PROPS}
      {...activationProps}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
    >
      {item.instance ? (
        <WidgetRendererById instanceId={item.id} widget={item.widget} presentation="compact" region="bottom" />
      ) : null}
    </Row>
  );
  // The Row is the popover's actual trigger (toggle, dismissal exemption,
  // focus restore, aria) while the outer Box keeps the tooltip's trigger id —
  // two elements, so the two machines never fight over one `id`.
  const content = (
    <Box
      ref={bindWrapper}
      display="flex"
      role="group"
      aria-label={item.label}
      tabIndex={-1}
      focusVisibleRing="inside"
      data-status-bar-widget={item.id}
      data-workbench-focus-preserve={item.widget.manifest.preserveWorkbenchFocus || undefined}
      h="full"
      style={style}
    >
      <WidgetRecoveryAnnouncement
        failed={recovery.hasFailure}
        generation={recovery.generation}
        retrying={recovery.retrying}
        label={item.label}
      />
      {item.isPopover ? (
        <Popover.Trigger asChild disabled={!!recovery.error}>
          {row}
        </Popover.Trigger>
      ) : (
        row
      )}
      {recovery.error && (item.isExpandable || item.isPopover) ? failureOpenButton : null}
      {!recovery.error && tooltipError ? (
        <CompactWidgetFailure
          isRetrying={recovery.retrying || recovery.reloading}
          label={item.label}
          message={tooltipError.message}
          onRetry={recovery.retry}
          onRequestReload={recovery.reloadAvailable ? recovery.requestReload : undefined}
        />
      ) : null}
    </Box>
  );
  // Recovery owns error details; retaining the outer root preserves the row's focus when it finishes.
  const labelTooltip = (
    <Tooltip
      closeDelay={80}
      disabled={!!recovery.error}
      keepsRootWhenDisabled
      content={item.failureMessage ? `${item.label}: ${item.failureMessage}` : item.label}
      openDelay={250}
      positioning={TOOLTIP_POSITIONING}
    >
      {content}
    </Tooltip>
  );

  if (item.isPopover) {
    return (
      <Popover.Root
        finalFocusEl={item.widget.manifest.preserveWorkbenchFocus ? restoreShortcutFocus : statusFocusTarget}
        lazyMount
        open={isPopoverOpen}
        positioning={WIDGET_POPOVER_POSITIONING}
        unmountOnExit
        onOpenChange={handlePopoverOpenChange}
      >
        {labelTooltip}
        <Portal>
          <Popover.Positioner>
            <PopoverContent
              data-workbench-focus-preserve={item.widget.manifest.preserveWorkbenchFocus || undefined}
              display="flex"
              flexDirection="column"
              maxH="min(28rem, var(--available-height))"
              maxW="calc(100vw - 16px)"
              p="0"
              w="26rem"
            >
              {/* Clipping lives one level down: `overflow: hidden` on the
                  content itself would cut off the anchor arrow. */}
              <Box display="flex" flex="1" flexDirection="column" minH="0" overflow="hidden" rounded="inherit">
                {item.instance ? (
                  <WidgetRendererById
                    instanceId={item.id}
                    widget={item.widget}
                    presentation="expanded"
                    region="popover"
                  />
                ) : null}
              </Box>
            </PopoverContent>
          </Popover.Positioner>
        </Portal>
      </Popover.Root>
    );
  }

  if (item.isExpandable) {
    return labelTooltip;
  }

  return (
    <Tooltip
      closeDelay={80}
      disabled={!!recovery.error}
      keepsRootWhenDisabled
      content={tooltipContent}
      openDelay={250}
      positioning={TOOLTIP_POSITIONING}
    >
      {content}
    </Tooltip>
  );
};

const BottomWidgetTooltipContent = ({ item }: { item: BottomWidgetItem }) => (
  <WidgetRendererById instanceId={item.id} widget={item.widget} presentation="tooltip" region="bottom" />
);
