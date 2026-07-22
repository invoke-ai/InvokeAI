import type { WidgetInstanceId } from '@workbench/widgetContracts';
import type { WidgetRegionDropState } from '@workbench/widgetDnd';
import type { PlacedWidgetRegionItem, WidgetPlacementInstanceMeta } from '@workbench/widgetRegionViewModel';

import { Box } from '@chakra-ui/react';
import { horizontalListSortingStrategy } from '@dnd-kit/sortable';
import { Row, Tooltip } from '@platform/ui';
import {
  WidgetEnableMenu,
  WidgetInstanceContextMenu,
  WidgetRendererById,
  WidgetStrip,
  useWidgetSortable,
  type WidgetEnableMenuItem,
  type WidgetInstanceContextMenuTarget,
} from '@workbench/widget-frame';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { closeWidgetPlacement, openWidgetPlacement, revealWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import {
  createWidgetRegionViewModelFromState,
  getWidgetRegionItems,
  isCompactBottomItem,
  isExpandableBottomItem,
} from '@workbench/widgetRegionViewModel';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { type KeyboardEvent, type MouseEvent, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface BottomWidgetItem extends PlacedWidgetRegionItem<WidgetPlacementInstanceMeta> {
  isExpandable: boolean;
}

const BOTTOM_MENU_POSITIONING = { placement: 'top-end' } as const;
const BOTTOM_MENU_TRIGGER = { kind: 'bottom' } as const;
const COMPACT_ROW_HOVER_PROPS = { color: 'fg' };
const COMPACT_ROW_ACTIVE_HOVER_PROPS = { color: 'accent.contrast' };
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

    return [{ ...item, isExpandable: isExpandableBottomItem(item) }];
  });
  const sortableInstanceIds = useMemo(() => compactItems.map((item) => item.id), [compactItems]);
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
  const handleContextClose = useCallback(() => setEnableMenuTarget(null), []);
  const handleInstanceClose = useCallback(() => setInstanceMenuTarget(null), []);

  return (
    <WidgetStrip
      align="center"
      as="footer"
      bg="bg.subtle"
      borderTopWidth="1px"
      borderColor="border.subtle"
      color="fg.muted"
      dropState={dropState}
      flexShrink={0}
      h="6"
      px="2"
      region="bottom"
      sortableInstanceIds={sortableInstanceIds}
      strategy={horizontalListSortingStrategy}
      w="full"
      onContextMenu={openEnableMenu}
    >
      {compactItems.map((item) => (
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
      <WidgetInstanceContextMenu
        target={instanceMenuTarget}
        onClose={handleInstanceClose}
        onRemove={toggleBottomWidget}
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
  const rowDragHandleProps = item.isExpandable
    ? Object.fromEntries(Object.entries(dragHandleProps).filter(([key]) => key !== 'onKeyDown'))
    : dragHandleProps;
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onSelect(item.id);
      }
    },
    [item.id, onSelect]
  );
  const activationProps = useMemo(
    () =>
      item.isExpandable
        ? {
            role: 'button' as const,
            tabIndex: 0,
            onKeyDown: handleKeyDown,
          }
        : {},
    [handleKeyDown, item.isExpandable]
  );
  const handleClick = useCallback(() => {
    if (item.isExpandable) {
      onSelect(item.id);
    }
  }, [item.id, item.isExpandable, onSelect]);
  const handleContextMenu = useCallback((event: MouseEvent) => onContextMenu(item, event), [item, onContextMenu]);
  const tooltipContent = useMemo(() => (item.instance ? <BottomWidgetTooltipContent item={item} /> : null), [item]);

  const content = (
    <Box ref={setNodeRef} h="full" style={style}>
      <Row
        {...rowDragHandleProps}
        active={isActive ? 'accent' : 'none'}
        aria-label={item.label}
        aria-pressed={isActive}
        cursor={isDragging ? 'grabbing' : item.isExpandable ? 'pointer' : 'default'}
        h="full"
        w="auto"
        _hover={isActive ? COMPACT_ROW_ACTIVE_HOVER_PROPS : COMPACT_ROW_HOVER_PROPS}
        {...activationProps}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
      >
        {item.instance ? (
          <WidgetRendererById instanceId={item.id} widget={item.widget} presentation="compact" region="bottom" />
        ) : null}
      </Row>
    </Box>
  );
  if (item.isExpandable) {
    return (
      <Tooltip
        closeDelay={80}
        content={item.failureMessage ? `${item.label}: ${item.failureMessage}` : item.label}
        openDelay={250}
        positioning={TOOLTIP_POSITIONING}
      >
        {content}
      </Tooltip>
    );
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
