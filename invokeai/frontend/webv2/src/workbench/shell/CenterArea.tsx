import type { WidgetRegionDropState } from '@workbench/widgetDnd';
import type {
  PlacedWidgetRegionItem,
  WidgetPlacementInstanceMeta,
  WidgetRegionItem,
} from '@workbench/widgetRegionViewModel';

import { Box, Flex, Text } from '@chakra-ui/react';
import { horizontalListSortingStrategy } from '@dnd-kit/sortable';
import { Tabs } from '@workbench/components/ui';
import { useFocusRegionProps } from '@workbench/focusRegions';
import { WidgetIcon } from '@workbench/iconResolver';
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
  isRequiredCenterView,
} from '@workbench/widgetRegionViewModel';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { type MouseEvent, useState } from 'react';

type CenterWidgetItem = PlacedWidgetRegionItem<WidgetPlacementInstanceMeta>;

/** Center work area: the view tab strip plus the active registered center view. */
export const CenterArea = ({ dropState }: { dropState: WidgetRegionDropState }) => {
  const placementProject = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const centerRegion = useActiveProjectSelector((project) => project.widgetRegions.center);
  const dispatch = useWorkbenchDispatch();
  const [enableMenuTarget, setEnableMenuTarget] = useState<{ x: number; y: number } | null>(null);
  const [instanceMenuTarget, setInstanceMenuTarget] = useState<WidgetInstanceContextMenuTarget | null>(null);
  const focusRegionProps = useFocusRegionProps('center');
  const centerRegionViewModel = createWidgetRegionViewModelFromState({
    region: 'center',
    regionState: centerRegion,
    widgetInstances: placementProject.widgetInstances,
    widgets: getWidgetsForRegion('center'),
  });
  const centerWidgetMenuItems = getWidgetRegionItems(centerRegionViewModel);
  const enabledCenterWidgetItems = centerRegionViewModel.placedItems.filter((item) => item.status === 'enabled');
  const centerViewItems = enabledCenterWidgetItems.filter((item) => item.widget.manifest.centerPlacement !== 'toolbar');
  const centerToolbarItems = enabledCenterWidgetItems.filter(
    (item) => item.widget.manifest.centerPlacement === 'toolbar'
  );
  const activeCenterViewId = centerViewItems.some((item) => item.id === centerRegion.activeInstanceId)
    ? centerRegion.activeInstanceId
    : centerViewItems[0]?.id;
  const openEnableMenu = (event: MouseEvent) => {
    event.preventDefault();
    setEnableMenuTarget({ x: event.clientX, y: event.clientY });
  };
  const openInstanceMenu = (item: CenterWidgetItem, event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setInstanceMenuTarget({ item, x: event.clientX, y: event.clientY });
  };
  const toggleCenterWidget = (item: WidgetEnableMenuItem) =>
    item.isEnabled
      ? closeWidgetPlacement({
          dispatch,
          getWidgetById,
          instanceId: item.id,
          project: placementProject,
          region: 'center',
        })
      : openWidgetPlacement({
          dispatch,
          getWidgetsForRegion,
          options: { createNew: item.allowMultiple, preferredRegions: ['center'] },
          typeId: item.typeId,
        });

  return (
    <Flex as="section" bg="bg" direction="column" flex="1" minH="0" minW="0" {...focusRegionProps}>
      <WidgetStrip
        align="center"
        bg="bg.subtle"
        borderBottomWidth="1px"
        borderColor="border.subtle"
        dropState={dropState}
        h="10"
        px="1.5"
        region="center"
        sortableInstanceIds={centerRegionViewModel.sortableInstanceIds.filter((instanceId) =>
          centerViewItems.some((item) => item.id === instanceId)
        )}
        strategy={horizontalListSortingStrategy}
        onContextMenu={openEnableMenu}
      >
        <Tabs.Root
          value={activeCenterViewId}
          h="full"
          w="full"
          onValueChange={(event) =>
            revealWidgetPlacement({
              dispatch,
              instanceId: event.value as CenterWidgetItem['id'],
              project: placementProject,
              region: 'center',
            })
          }
        >
          <Tabs.List>
            {centerViewItems.map((item) => (
              <SortableCenterTab key={item.id} item={item} onContextMenu={openInstanceMenu} />
            ))}
          </Tabs.List>
        </Tabs.Root>
        <Box flex="1" />
        {centerToolbarItems.map((item) => (
          <Flex key={item.id} align="center" h="full">
            <WidgetRendererById instanceId={item.id} widget={item.widget} presentation="compact" region="center" />
          </Flex>
        ))}
        <WidgetEnableMenu
          contextTarget={enableMenuTarget}
          getItemMeta={(item) =>
            isRequiredCenterView(item as WidgetRegionItem, centerViewItems.length) ? 'Required' : null
          }
          groupLabel="Center Widgets"
          isItemDisabled={(item) => isRequiredCenterView(item as WidgetRegionItem, centerViewItems.length)}
          items={centerWidgetMenuItems}
          positioning={{ placement: 'bottom-end' }}
          trigger={{ kind: 'center' }}
          triggerLabel="Center widget menu"
          onContextClose={() => setEnableMenuTarget(null)}
          onToggle={toggleCenterWidget}
        />
        <WidgetInstanceContextMenu
          isRemoveDisabled={(item) => isRequiredCenterView(item as WidgetRegionItem, centerViewItems.length)}
          target={instanceMenuTarget}
          onClose={() => setInstanceMenuTarget(null)}
          onRemove={toggleCenterWidget}
        />
      </WidgetStrip>

      <Box flex="1" minH="0" position="relative">
        <CenterViewSlot activeWidgetId={activeCenterViewId} items={centerViewItems} />
      </Box>
    </Flex>
  );
};

const SortableCenterTab = ({
  item,
  onContextMenu,
}: {
  item: CenterWidgetItem;
  onContextMenu: (item: CenterWidgetItem, event: MouseEvent) => void;
}) => {
  const { dragHandleProps, setNodeRef, style } = useWidgetSortable({
    disabled: item.status === 'disabled',
    instanceId: item.id,
    region: 'center',
    typeId: item.instance.typeId,
  });

  return (
    <Tabs.Trigger
      ref={setNodeRef}
      value={item.id}
      disabled={item.status === 'disabled'}
      fontSize="xs"
      px="3"
      style={style}
      {...dragHandleProps}
      onContextMenu={(event) => onContextMenu(item, event)}
    >
      <WidgetIcon icon={item.widget.manifest.icon} boxSize="3.5" />
      {item.label}
    </Tabs.Trigger>
  );
};

const CenterViewSlot = ({
  activeWidgetId,
  items,
}: {
  activeWidgetId: string | undefined;
  items: CenterWidgetItem[];
}) => {
  const widget = items.find((item) => item.id === activeWidgetId)?.widget;
  const instance = items.find((item) => item.id === activeWidgetId)?.instance;
  const View = widget?.manifest.view;

  if (!instance || !widget || widget.status !== 'enabled' || !View) {
    return <FallbackCenterView label="Center widget unavailable" />;
  }

  return <WidgetRendererById instanceId={instance.id} widget={widget} region="center" />;
};

const FallbackCenterView = ({ label }: { label: string }) => (
  <Flex align="center" h="full" justify="center" w="full">
    <Text color="fg.subtle" fontSize="sm" textTransform="capitalize">
      {label} view
    </Text>
  </Flex>
);
