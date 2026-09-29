import type { WidgetInstanceId } from '@workbench/widgetContracts';
import type { WidgetRegionDropState } from '@workbench/widgetDnd';
import type { PlacedWidgetRegionItem, WidgetPlacementInstanceMeta } from '@workbench/widgetRegionViewModel';

import { Box, Flex, Icon, Popover, Portal } from '@chakra-ui/react';
import { useDndContext, useDroppable } from '@dnd-kit/core';
import { horizontalListSortingStrategy } from '@dnd-kit/sortable';
import { PopoverContent, Row, Tooltip } from '@platform/ui';
import { useHighlightedRegion } from '@workbench/focusRegions';
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
import { type KeyboardEvent, type MouseEvent, type ReactNode, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface BottomWidgetItem extends PlacedWidgetRegionItem<WidgetPlacementInstanceMeta> {
  isExpandable: boolean;
  isPopover: boolean;
}

const BOTTOM_MENU_POSITIONING = { placement: 'top-end' } as const;
const WIDGET_POPOVER_POSITIONING = { placement: 'top-end' } as const;
const BOTTOM_MENU_TRIGGER = { kind: 'bottom' } as const;
/** Use content color for active brand accents; background tint is indistinguishable on the light theme. */
const COMPACT_ROW_HOVER_PROPS = { bg: 'bg.emphasized', color: 'fg' };

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
    <Flex ref={setNodeRef} align="center" alignSelf="stretch" flexShrink={0} position="relative">
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
      <Flex align="center" alignSelf="stretch" flexShrink={0} position="relative">
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

      <Box flex="1" />
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
  const { dragHandleProps, isDragging, setNodeRef, style } = useWidgetSortable({
    instanceId: item.id,
    region: 'bottom',
    typeId: item.typeId,
  });
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);
  const isActivatable = item.isExpandable || item.isPopover;
  const rowDragHandleProps = isActivatable
    ? Object.fromEntries(Object.entries(dragHandleProps).filter(([key]) => key !== 'onKeyDown'))
    : dragHandleProps;
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLElement>) => {
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
    [item.id, item.isPopover, onSelect]
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
  const handleClick = useCallback(() => {
    if (item.isExpandable) {
      onSelect(item.id);
    }
  }, [item.id, item.isExpandable, onSelect]);
  const handleContextMenu = useCallback((event: MouseEvent) => onContextMenu(item, event), [item, onContextMenu]);
  const handlePopoverOpenChange = useCallback((event: { open: boolean }) => setIsPopoverOpen(event.open), []);
  const tooltipContent = useMemo(() => (item.instance ? <BottomWidgetTooltipContent item={item} /> : null), [item]);
  const isRowActive = item.isPopover ? isPopoverOpen : isActive;

  const row = (
    <Row
      {...rowDragHandleProps}
      aria-label={item.label}
      aria-pressed={item.isPopover ? undefined : isRowActive}
      color={isRowActive ? undefined : 'fg.muted'}
      cursor={isDragging ? 'grabbing' : 'default'}
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
    <Box ref={setNodeRef} h="full" style={style}>
      {item.isPopover ? <Popover.Trigger asChild>{row}</Popover.Trigger> : row}
    </Box>
  );
  const labelTooltip = (
    <Tooltip
      closeDelay={80}
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
              display="flex"
              flexDirection="column"
              maxH="min(28rem, var(--available-height))"
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
    <Tooltip closeDelay={80} content={tooltipContent} openDelay={250} positioning={TOOLTIP_POSITIONING}>
      {content}
    </Tooltip>
  );
};

const BottomWidgetTooltipContent = ({ item }: { item: BottomWidgetItem }) => (
  <WidgetRendererById instanceId={item.id} widget={item.widget} presentation="tooltip" region="bottom" />
);
