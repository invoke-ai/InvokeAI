import type { WidgetRegionDropState } from '@workbench/widgetDnd';
import type {
  PlacedWidgetRegionItem,
  WidgetPlacementInstanceMeta,
  WidgetRegionItem,
} from '@workbench/widgetRegionViewModel';

import { Box, Flex, Text } from '@chakra-ui/react';
import { horizontalListSortingStrategy } from '@dnd-kit/sortable';
import { useModelLoads } from '@features/models';
import { getProjectQueueIndicatorState, type QueueProgressBarState } from '@features/queue/contracts';
import { useQueueItemProgress } from '@features/queue/react';
import { Tabs } from '@platform/ui';
import { QueueTabBackgroundProgress } from '@workbench/components/QueueProgressIndicator';
import { useFocusRegionProps } from '@workbench/focusRegions';
import { WidgetIcon } from '@workbench/iconResolver';
import {
  WidgetEnableMenu,
  WidgetInstanceContextMenu,
  WidgetRendererById,
  WidgetStrip,
  useWidgetIntentPreloadProps,
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
  isRequiredCenterView,
} from '@workbench/widgetRegionViewModel';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { type MouseEvent, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

type CenterWidgetItem = PlacedWidgetRegionItem<WidgetPlacementInstanceMeta>;

const CENTER_MENU_POSITIONING = { placement: 'bottom-end' } as const;
const CENTER_MENU_TRIGGER = { kind: 'center' } as const;
const CENTER_PREFERRED_REGIONS = ['center'] as const;

/** Center work area: the view tab strip plus the active registered center view. */
export const CenterArea = ({ dropState }: { dropState: WidgetRegionDropState }) => {
  const { t } = useTranslation();
  const placementProject = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const centerRegion = useActiveProjectSelector((project) => project.widgetRegions.center);
  const invocation = useActiveProjectSelector((project) => project.invocation);
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status);
  const modelLoads = useModelLoads();
  const { widgets } = useWorkbenchCommands();
  const [enableMenuTarget, setEnableMenuTarget] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [instanceMenuTarget, setInstanceMenuTarget] = useState<WidgetInstanceContextMenuTarget | null>(null);
  const focusRegionProps = useFocusRegionProps('center');
  const getWidgetLabel = useCallback(
    (manifest: Parameters<typeof resolveWidgetLabel>[0]) => resolveWidgetLabel(manifest, t),
    [t]
  );

  const centerRegionViewModel = createWidgetRegionViewModelFromState({
    getWidgetLabel,
    region: 'center',
    regionState: centerRegion,
    widgetInstances: placementProject.widgetInstances,
    widgets: getWidgetsForRegion('center'),
  });

  const centerWidgetMenuItems = useMemo(() => getWidgetRegionItems(centerRegionViewModel), [centerRegionViewModel]);

  const enabledCenterWidgetItems = useMemo(
    () => centerRegionViewModel.placedItems.filter((item) => item.status === 'enabled'),
    [centerRegionViewModel.placedItems]
  );

  const centerViewItems = useMemo(
    () => enabledCenterWidgetItems.filter((item) => item.widget.manifest.centerPlacement !== 'toolbar'),
    [enabledCenterWidgetItems]
  );

  const centerToolbarItems = useMemo(
    () => enabledCenterWidgetItems.filter((item) => item.widget.manifest.centerPlacement === 'toolbar'),
    [enabledCenterWidgetItems]
  );

  const activeCenterViewId = centerViewItems.some((item) => item.id === centerRegion.activeInstanceId)
    ? centerRegion.activeInstanceId
    : centerViewItems[0]?.id;

  const centerSortableInstanceIds = useMemo(
    () =>
      centerRegionViewModel.sortableInstanceIds.filter((instanceId) =>
        centerViewItems.some((item) => item.id === instanceId)
      ),
    [centerRegionViewModel.sortableInstanceIds, centerViewItems]
  );

  const openEnableMenu = useCallback((event: MouseEvent) => {
    event.preventDefault();
    setEnableMenuTarget({ x: event.clientX, y: event.clientY });
  }, []);

  const openInstanceMenu = useCallback((item: CenterWidgetItem, event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setInstanceMenuTarget({ item, x: event.clientX, y: event.clientY });
  }, []);

  const toggleCenterWidget = useCallback(
    (item: WidgetEnableMenuItem) =>
      item.isEnabled
        ? closeWidgetPlacement({
            widgets,
            getWidgetById,
            instanceId: item.id,
            project: placementProject,
            region: 'center',
          })
        : openWidgetPlacement({
            widgets,
            getWidgetsForRegion,
            options: {
              createNew: item.allowMultiple,
              preferredRegions: CENTER_PREFERRED_REGIONS,
            },
            typeId: item.typeId,
          }),
    [placementProject, widgets]
  );

  const handleCenterTabChange = useCallback(
    (event: { value: string }) =>
      revealWidgetPlacement({
        instanceId: event.value as CenterWidgetItem['id'],
        project: placementProject,
        region: 'center',
        widgets,
      }),
    [placementProject, widgets]
  );

  const handleContextClose = useCallback(() => setEnableMenuTarget(null), []);
  const handleInstanceClose = useCallback(() => setInstanceMenuTarget(null), []);

  const getItemMeta = useCallback(
    (item: WidgetEnableMenuItem) =>
      isRequiredCenterView(item as WidgetRegionItem, centerViewItems.length) ? 'Required' : null,
    [centerViewItems.length]
  );

  const isItemDisabled = useCallback(
    (item: WidgetEnableMenuItem) => isRequiredCenterView(item as WidgetRegionItem, centerViewItems.length),
    [centerViewItems.length]
  );

  const baseIndicatorState = getProjectQueueIndicatorState({
    isConnected: backendConnectionStatus === 'connected',
    loadingModelsCount: modelLoads.length,
    progress: null,
    queueItems,
  });

  const progress = useQueueItemProgress(baseIndicatorState.runningQueueItemId ?? '');

  const indicatorState = getProjectQueueIndicatorState({
    isConnected: backendConnectionStatus === 'connected',
    loadingModelsCount: modelLoads.length,
    progress,
    queueItems,
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
        sortableInstanceIds={centerSortableInstanceIds}
        strategy={horizontalListSortingStrategy}
        onContextMenu={openEnableMenu}
      >
        <Tabs.Root value={activeCenterViewId} h="full" w="full" onValueChange={handleCenterTabChange}>
          <Tabs.List>
            {centerViewItems.map((item) => (
              <SortableCenterTab
                key={item.id}
                item={item}
                progressState={indicatorState.progressState}
                showProgress={indicatorState.hasOpenQueueWork && item.typeId === invocation.sourceId}
                onContextMenu={openInstanceMenu}
              />
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
          getItemMeta={getItemMeta}
          groupLabel="Center Widgets"
          isItemDisabled={isItemDisabled}
          items={centerWidgetMenuItems}
          positioning={CENTER_MENU_POSITIONING}
          trigger={CENTER_MENU_TRIGGER}
          triggerLabel="Center widget menu"
          onContextClose={handleContextClose}
          onToggle={toggleCenterWidget}
        />

        <WidgetInstanceContextMenu
          isRemoveDisabled={isItemDisabled}
          target={instanceMenuTarget}
          onClose={handleInstanceClose}
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
  progressState,
  showProgress,
}: {
  item: CenterWidgetItem;
  onContextMenu: (item: CenterWidgetItem, event: MouseEvent) => void;
  progressState: QueueProgressBarState;
  showProgress: boolean;
}) => {
  const { dragHandleProps, setNodeRef, style } = useWidgetSortable({
    disabled: item.status === 'disabled',
    instanceId: item.id,
    region: 'center',
    typeId: item.instance.typeId,
  });
  const handleContextMenu = useCallback((event: MouseEvent) => onContextMenu(item, event), [item, onContextMenu]);
  const intentPreloadProps = useWidgetIntentPreloadProps(item.widget, item.status === 'disabled');

  return (
    <Tabs.Trigger
      ref={setNodeRef}
      value={item.id}
      disabled={item.status === 'disabled'}
      fontSize="xs"
      overflow="hidden"
      position="relative"
      px="3"
      style={style}
      {...dragHandleProps}
      {...intentPreloadProps}
      onContextMenu={handleContextMenu}
    >
      {showProgress ? <QueueTabBackgroundProgress state={progressState} zIndex="-1" /> : null}

      <Box alignItems="center" display="inline-flex" gap="2" position="relative" zIndex="1">
        <WidgetIcon icon={item.widget.manifest.icon} boxSize="3.5" />
        {item.label}
      </Box>
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
  if (!instance || !widget || widget.status !== 'enabled') {
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
