import type { WidgetInstanceId } from '@workbench/types';
import type { WidgetRegionDropState } from '@workbench/widgetDnd';
import type { PlacedWidgetRegionItem, WidgetPlacementInstanceMeta } from '@workbench/widgetRegionViewModel';

import { Box } from '@chakra-ui/react';
import { horizontalListSortingStrategy } from '@dnd-kit/sortable';
import { Row, Tooltip } from '@workbench/components/ui';
import {
  WidgetEnableMenu,
  WidgetInstanceContextMenu,
  WidgetRendererById,
  WidgetStrip,
  useWidgetSortable,
  type WidgetEnableMenuItem,
  type WidgetInstanceContextMenuTarget,
} from '@workbench/widget-frame';
import { closeWidgetPlacement, openWidgetPlacement, revealWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import {
  createWidgetRegionViewModelFromState,
  getWidgetRegionItems,
  isCompactBottomItem,
  isExpandableBottomItem,
} from '@workbench/widgetRegionViewModel';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { type KeyboardEvent, type MouseEvent, useState } from 'react';

interface BottomWidgetItem extends PlacedWidgetRegionItem<WidgetPlacementInstanceMeta> {
  isExpandable: boolean;
}

export const StatusBar = ({ dropState }: { dropState: WidgetRegionDropState }) => {
  const placementProject = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const bottomRegion = useActiveProjectSelector((project) => project.widgetRegions.bottom);
  const dispatch = useWorkbenchDispatch();
  const [enableMenuTarget, setEnableMenuTarget] = useState<{ x: number; y: number } | null>(null);
  const [instanceMenuTarget, setInstanceMenuTarget] = useState<WidgetInstanceContextMenuTarget | null>(null);
  const bottomRegionViewModel = createWidgetRegionViewModelFromState({
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
  const openEnableMenu = (event: MouseEvent) => {
    event.preventDefault();
    setEnableMenuTarget({ x: event.clientX, y: event.clientY });
  };
  const openInstanceMenu = (item: BottomWidgetItem, event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setInstanceMenuTarget({ item, x: event.clientX, y: event.clientY });
  };
  const toggleBottomWidget = (item: WidgetEnableMenuItem) =>
    item.isEnabled
      ? closeWidgetPlacement({
          dispatch,
          getWidgetById,
          instanceId: item.id,
          project: placementProject,
          region: 'bottom',
        })
      : openWidgetPlacement({
          dispatch,
          getWidgetsForRegion,
          options: { createNew: item.allowMultiple, preferredRegions: ['bottom'] },
          typeId: item.typeId,
        });

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
      sortableInstanceIds={compactItems.map((item) => item.id)}
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
          onSelect={(instanceId) =>
            revealWidgetPlacement({ dispatch, instanceId, project: placementProject, region: 'bottom' })
          }
        />
      ))}

      <Box flex="1" />

      <WidgetEnableMenu
        contextTarget={enableMenuTarget}
        groupLabel="Bottom Widgets"
        items={items}
        positioning={{ placement: 'top-end' }}
        trigger={{ kind: 'bottom' }}
        triggerLabel="Bottom widget visibility"
        onContextClose={() => setEnableMenuTarget(null)}
        onToggle={toggleBottomWidget}
      />
      <WidgetInstanceContextMenu
        target={instanceMenuTarget}
        onClose={() => setInstanceMenuTarget(null)}
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
  const View = item.widget.manifest.view;
  const { dragHandleProps, isDragging, setNodeRef, style } = useWidgetSortable({
    instanceId: item.id,
    region: 'bottom',
    typeId: item.typeId,
  });
  const rowDragHandleProps = item.isExpandable
    ? Object.fromEntries(Object.entries(dragHandleProps).filter(([key]) => key !== 'onKeyDown'))
    : dragHandleProps;
  const activationProps = item.isExpandable
    ? {
        role: 'button' as const,
        tabIndex: 0,
        onKeyDown: (event: KeyboardEvent) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onSelect(item.id);
          }
        },
      }
    : {};

  if (!View) {
    return null;
  }

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
        _hover={{ color: isActive ? 'accent.contrast' : 'fg' }}
        {...activationProps}
        onClick={() => {
          if (item.isExpandable) {
            onSelect(item.id);
          }
        }}
        onContextMenu={(event) => onContextMenu(item, event)}
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
        positioning={{ placement: 'top-start' }}
      >
        {content}
      </Tooltip>
    );
  }

  return (
    <Tooltip
      closeDelay={80}
      content={
        item.instance ? (
          <WidgetRendererById instanceId={item.id} widget={item.widget} presentation="tooltip" region="bottom" />
        ) : null
      }
      openDelay={250}
      positioning={{ placement: 'top-start' }}
    >
      {content}
    </Tooltip>
  );
};
