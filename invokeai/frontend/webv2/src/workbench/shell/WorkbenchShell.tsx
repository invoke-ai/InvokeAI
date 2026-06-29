import { Flex, HStack, Text } from '@chakra-ui/react';
import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragStartEvent,
} from '@dnd-kit/core';
import { restrictToWindowEdges } from '@dnd-kit/modifiers';
import { sortableKeyboardCoordinates } from '@dnd-kit/sortable';
import { FocusRegionProvider } from '@workbench/focusRegions';
import { WidgetIcon } from '@workbench/iconResolver';
import { WidgetBar } from '@workbench/widget-frame';
import {
  getRegionDropState,
  isWidgetDndData,
  isWidgetInstanceDragData,
  resolveWidgetDragEnd,
  type ActiveWidgetDrag,
  widgetCollisionDetection,
} from '@workbench/widgetDnd';
import {
  closeWidgetPlacement,
  dispatchWidgetDragEndPlacement,
  openWidgetPlacement,
  revealWidgetPlacement,
} from '@workbench/widgetPlacementCommands';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { createWidgetRegionViewModelFromState, getWidgetRegionItems } from '@workbench/widgetRegionViewModel';
import { getWidgetById, getWidgetsForRegion, widgetRegistrationFailures } from '@workbench/widgetRegistry';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { WorkbenchWidgetRegistryProvider } from '@workbench/WorkbenchWidgetRegistryContext';
import { useCallback, useEffect, useMemo, useState } from 'react';

import { BottomPanel } from './BottomPanel';
import { CenterArea } from './CenterArea';
import { WorkbenchNotificationToaster } from './notifications';
import { LeftPanel, RightPanel } from './Panels';
import { StatusBar } from './StatusBar';
import { TopBar } from './topbar';

const DND_MODIFIERS = [restrictToWindowEdges];

export const WorkbenchShell = () => {
  const dispatch = useWorkbenchDispatch();
  const panels = useActiveProjectSelector((project) => project.layout.panels);
  const leftRegion = useActiveProjectSelector((project) => project.widgetRegions.left);
  const rightRegion = useActiveProjectSelector((project) => project.widgetRegions.right);
  const placementProject = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );
  const [activeDrag, setActiveDrag] = useState<ActiveWidgetDrag | null>(null);
  const leftRegionViewModel = useMemo(
    () =>
      createWidgetRegionViewModelFromState({
        region: 'left',
        regionState: leftRegion,
        widgetInstances: placementProject.widgetInstances,
        widgets: getWidgetsForRegion('left'),
      }),
    [leftRegion, placementProject.widgetInstances]
  );
  const rightRegionViewModel = useMemo(
    () =>
      createWidgetRegionViewModelFromState({
        region: 'right',
        regionState: rightRegion,
        widgetInstances: placementProject.widgetInstances,
        widgets: getWidgetsForRegion('right'),
      }),
    [rightRegion, placementProject.widgetInstances]
  );
  const leftMenuItems = useMemo(() => getWidgetRegionItems(leftRegionViewModel), [leftRegionViewModel]);
  const rightMenuItems = useMemo(() => getWidgetRegionItems(rightRegionViewModel), [rightRegionViewModel]);
  const leftRailItems = useMemo(
    () => leftRegionViewModel.placedItems.filter((item) => item.status !== 'disabled'),
    [leftRegionViewModel]
  );
  const rightRailItems = useMemo(
    () => rightRegionViewModel.placedItems.filter((item) => item.status !== 'disabled'),
    [rightRegionViewModel]
  );
  const canShowLeftPanel = leftRailItems.some((item) => item.id === leftRegion.activeInstanceId);
  const canShowRightPanel = rightRailItems.some((item) => item.id === rightRegion.activeInstanceId);
  const leftDropState = useMemo(
    () => getRegionDropState(placementProject, activeDrag, 'left', getWidgetById),
    [activeDrag, placementProject]
  );
  const rightDropState = useMemo(
    () => getRegionDropState(placementProject, activeDrag, 'right', getWidgetById),
    [activeDrag, placementProject]
  );
  const centerDropState = useMemo(
    () => getRegionDropState(placementProject, activeDrag, 'center', getWidgetById),
    [activeDrag, placementProject]
  );
  const bottomDropState = useMemo(
    () => getRegionDropState(placementProject, activeDrag, 'bottom', getWidgetById),
    [activeDrag, placementProject]
  );

  useEffect(() => {
    for (const failure of widgetRegistrationFailures) {
      dispatch({ failure, type: 'recordWidgetFailure' });
    }
  }, [dispatch]);

  const handleDragStart = useCallback(
    (event: DragStartEvent) => {
      const activeData = event.active.data.current;

      if (!isWidgetInstanceDragData(activeData)) {
        return;
      }

      const instance = placementProject.widgetInstances[activeData.instanceId];
      const widget = instance ? getWidgetById(instance.typeId) : undefined;

      if (!instance || !widget) {
        return;
      }

      setActiveDrag({
        fromRegion: activeData.region,
        icon: widget.manifest.icon,
        instanceId: activeData.instanceId,
        label: instance.title ?? widget.manifest.labelText,
        typeId: instance.typeId,
      });
    },
    [placementProject.widgetInstances]
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const activeData = event.active.data.current;
      const overData = event.over?.data.current ?? null;

      setActiveDrag(null);

      if (!isWidgetInstanceDragData(activeData) || !isWidgetDndData(overData)) {
        return;
      }

      const resolution = resolveWidgetDragEnd(placementProject, activeData, overData, getWidgetById);

      if (!resolution) {
        return;
      }

      dispatchWidgetDragEndPlacement({ dispatch, resolution });
    },
    [dispatch, placementProject]
  );
  const handleDragCancel = useCallback(() => setActiveDrag(null), []);
  const handleSelectLeft = useCallback(
    (instanceId: string) => revealWidgetPlacement({ dispatch, instanceId, project: placementProject, region: 'left' }),
    [dispatch, placementProject]
  );
  const handleSelectRight = useCallback(
    (instanceId: string) => revealWidgetPlacement({ dispatch, instanceId, project: placementProject, region: 'right' }),
    [dispatch, placementProject]
  );
  const handleToggleLeft = useCallback(
    (item: (typeof leftMenuItems)[number]) =>
      item.isEnabled
        ? closeWidgetPlacement({
            dispatch,
            getWidgetById,
            instanceId: item.id,
            project: placementProject,
            region: 'left',
          })
        : openWidgetPlacement({
            dispatch,
            getWidgetsForRegion,
            options: { createNew: item.allowMultiple, preferredRegions: ['left'] },
            typeId: item.typeId,
          }),
    [dispatch, placementProject]
  );
  const handleToggleRight = useCallback(
    (item: (typeof rightMenuItems)[number]) =>
      item.isEnabled
        ? closeWidgetPlacement({
            dispatch,
            getWidgetById,
            instanceId: item.id,
            project: placementProject,
            region: 'right',
          })
        : openWidgetPlacement({
            dispatch,
            getWidgetsForRegion,
            options: { createNew: item.allowMultiple, preferredRegions: ['right'] },
            typeId: item.typeId,
          }),
    [dispatch, placementProject]
  );

  return (
    <WorkbenchWidgetRegistryProvider getWidgetById={getWidgetById} getWidgetsForRegion={getWidgetsForRegion}>
      <FocusRegionProvider>
        <DndContext
          collisionDetection={widgetCollisionDetection}
          modifiers={DND_MODIFIERS}
          sensors={sensors}
          onDragCancel={handleDragCancel}
          onDragEnd={handleDragEnd}
          onDragStart={handleDragStart}
        >
          <Flex direction="column" h="100vh" w="100vw">
            <WorkbenchNotificationToaster />
            <TopBar />

            <Flex as="main" flex="1" minH="0" overflow="hidden">
              <WidgetBar
                activeId={panels.isLeftOpen && !leftRegion.isCollapsed ? leftRegion.activeInstanceId : null}
                dropState={leftDropState}
                menuItems={leftMenuItems}
                railItems={leftRailItems}
                region="left"
                side="left"
                onSelect={handleSelectLeft}
                onToggle={handleToggleLeft}
              />
              {panels.isLeftOpen && !leftRegion.isCollapsed && canShowLeftPanel ? (
                <LeftPanel instanceId={leftRegion.activeInstanceId} />
              ) : null}
              <CenterArea dropState={centerDropState} />
              {panels.isRightOpen && !rightRegion.isCollapsed && canShowRightPanel ? (
                <RightPanel instanceId={rightRegion.activeInstanceId} />
              ) : null}
              <WidgetBar
                activeId={panels.isRightOpen && !rightRegion.isCollapsed ? rightRegion.activeInstanceId : null}
                dropState={rightDropState}
                menuItems={rightMenuItems}
                railItems={rightRailItems}
                region="right"
                side="right"
                onSelect={handleSelectRight}
                onToggle={handleToggleRight}
              />
            </Flex>

            <BottomPanel />
            <StatusBar dropState={bottomDropState} />
          </Flex>
          <DragOverlay>{activeDrag ? <WidgetDragPreview activeDrag={activeDrag} /> : null}</DragOverlay>
        </DndContext>
      </FocusRegionProvider>
    </WorkbenchWidgetRegistryProvider>
  );
};

const WidgetDragPreview = ({ activeDrag }: { activeDrag: ActiveWidgetDrag }) => (
  <HStack bg="bg" borderWidth="1px" gap="2" px="3" py="2" rounded="md" shadow="lg">
    <WidgetIcon icon={activeDrag.icon} boxSize="4" />
    <Text fontSize="xs" fontWeight="700">
      {activeDrag.label}
    </Text>
  </HStack>
);
