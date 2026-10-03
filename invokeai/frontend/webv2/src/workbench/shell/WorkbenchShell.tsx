import { Flex, HStack, Text, VisuallyHidden } from '@chakra-ui/react';
import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragStartEvent,
} from '@dnd-kit/core';
import { restrictToWindowEdges } from '@dnd-kit/modifiers';
import { sortableKeyboardCoordinates } from '@dnd-kit/sortable';
import { GalleryDragCursor, GalleryDragScope } from '@features/gallery/utility';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { WidgetIcon } from '@workbench/iconResolver';
import { PROJECT_CONTENT_PANEL_ID } from '@workbench/projects/projectTabsA11y';
import { WidgetBar, type WidgetBarGroup } from '@workbench/widget-frame';
import { FloatingWidgetLayer } from '@workbench/widget-frame/FloatingWidgetLayer';
import {
  getRegionDropState,
  isWidgetDndData,
  isWidgetInstanceDragData,
  resolveWidgetDragEnd,
  type ActiveWidgetDrag,
  workbenchAutoScroll,
  widgetCollisionDetection,
} from '@workbench/widgetDnd';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import {
  activateRailPlacement,
  closeWidgetPlacement,
  dispatchWidgetDragEndPlacement,
  dockFloatingPlacement,
  openWidgetPlacement,
  removeFloatingPlacement,
} from '@workbench/widgetPlacementCommands';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { createWidgetRegionViewModelFromState, getWidgetRegionItems } from '@workbench/widgetRegionViewModel';
import { getWidgetById, getWidgetsForRegion, widgetRegistrationFailures } from '@workbench/widgetRegistry';
import { useActiveProjectId, useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { BottomPanel } from './BottomPanel';
import { CenterArea } from './CenterArea';
import { DocumentTitleProgress } from './DocumentTitleProgress';
import {
  HoldToDragSensor,
  PrimaryMouseSensor,
  TOUCH_DRAG_HOLD_DELAY_MS,
  TOUCH_DRAG_MOVE_TOLERANCE_PX,
} from './holdToDragSensor';
import { WorkbenchNotificationToaster } from './notifications';
import { LeftPanel, RightPanel } from './Panels';
import { PasteMediaRuntime } from './PasteMediaRuntime';
import { ProjectConflictBanner } from './ProjectConflictBanner';
import { QueueRecoveryBanner } from './QueueRecoveryBanner';
import { StatusBar } from './StatusBar';
import { TopBar } from './topbar';

const DND_MODIFIERS = [restrictToWindowEdges];

export const WorkbenchShell = () => {
  const { notifications, widgets } = useWorkbenchCommands();
  const { focusFloating, focusRegion } = useWorkbenchFocus();
  const { t } = useTranslation();
  const panels = useActiveProjectSelector((project) => project.layout.panels);
  const projectId = useActiveProjectId();
  const projectName = useActiveProjectSelector((project) => project.name);
  const leftRegion = useActiveProjectSelector((project) => project.widgetRegions.left);
  const rightRegion = useActiveProjectSelector((project) => project.widgetRegions.right);
  // Placement only: a window's geometry, mode, and stacking change without re-rendering the shell or its rails.
  const placementProject = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const { floatingPlacements } = placementProject;
  const sensors = useSensors(
    useSensor(PrimaryMouseSensor, { activationConstraint: { distance: 6 } }),
    useSensor(HoldToDragSensor, {
      activationConstraint: { delay: TOUCH_DRAG_HOLD_DELAY_MS, tolerance: TOUCH_DRAG_MOVE_TOLERANCE_PX },
    }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );
  const [activeDrag, setActiveDrag] = useState<ActiveWidgetDrag | null>(null);
  const getWidgetLabel = useCallback(
    (manifest: Parameters<typeof resolveWidgetLabel>[0]) => resolveWidgetLabel(manifest, t),
    [t]
  );
  const leftRegionViewModel = useMemo(
    () =>
      createWidgetRegionViewModelFromState({
        floatingWidgets: floatingPlacements,
        region: 'left',
        regionState: leftRegion,
        widgetInstances: placementProject.widgetInstances,
        widgets: getWidgetsForRegion('left'),
        getWidgetLabel,
      }),
    [floatingPlacements, getWidgetLabel, leftRegion, placementProject.widgetInstances]
  );
  const rightRegionViewModel = useMemo(
    () =>
      createWidgetRegionViewModelFromState({
        floatingWidgets: floatingPlacements,
        region: 'right',
        regionState: rightRegion,
        widgetInstances: placementProject.widgetInstances,
        widgets: getWidgetsForRegion('right'),
        getWidgetLabel,
      }),
    [floatingPlacements, getWidgetLabel, placementProject.widgetInstances, rightRegion]
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
  const isLeftPanelShown = panels.isLeftOpen && !leftRegion.isCollapsed && canShowLeftPanel;
  const isRightPanelShown = panels.isRightOpen && !rightRegion.isCollapsed && canShowRightPanel;
  const leftDropState = useMemo(
    () => getRegionDropState(placementProject, activeDrag, 'left', getWidgetById),
    [activeDrag, placementProject]
  );
  const rightDropState = useMemo(
    () => getRegionDropState(placementProject, activeDrag, 'right', getWidgetById),
    [activeDrag, placementProject]
  );
  const bottomDropState = useMemo(
    () => getRegionDropState(placementProject, activeDrag, 'bottom', getWidgetById),
    [activeDrag, placementProject]
  );

  useMountEffect(() => {
    for (const failure of widgetRegistrationFailures) {
      notifications.recordWidgetFailure(failure);
    }
  });

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
        label: instance.title ?? getWidgetLabel(widget.manifest),
        typeId: instance.typeId,
      });
    },
    [getWidgetLabel, placementProject.widgetInstances]
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

      dispatchWidgetDragEndPlacement({ resolution, widgets });
    },
    [placementProject, widgets]
  );
  const handleDragCancel = useCallback(() => setActiveDrag(null), []);
  // Focus follows each of these: into the window a marker shows or the region a tab shows in, into the region a
  // window docks to, and into a center that gets its view back when the window is removed. A tab that collapses
  // its region shows nothing, and the move gives up.
  const handleSelect = useCallback(
    (region: WidgetBarGroup['region'], instanceId: string) => {
      const activated = activateRailPlacement({ instanceId, project: placementProject, region, widgets });

      if (activated === 'window') {
        focusFloating(instanceId);
      } else if (activated === 'tab') {
        focusRegion(region, placementProject.widgetInstances[instanceId]?.typeId);
      }
    },
    [focusFloating, focusRegion, placementProject, widgets]
  );
  const handleDock = useCallback(
    (instanceId: string) => {
      const typeId = placementProject.widgetInstances[instanceId]?.typeId;
      const returnRegion = dockFloatingPlacement({ instanceId, project: placementProject, widgets });

      if (returnRegion && typeId) {
        focusRegion(returnRegion, typeId);
      }
    },
    [focusRegion, placementProject, widgets]
  );
  const handleRemoveFloating = useCallback(
    (instanceId: string) => {
      const typeId = placementProject.widgetInstances[instanceId]?.typeId;
      const removed = removeFloatingPlacement({ instanceId, project: placementProject, widgets });

      if (!removed) {
        return;
      }

      // The marker's menu that asked and the marker itself are both gone. Focus goes to the center when it gets
      // its view back, otherwise to the panel the window's own rail is showing. Removing leaves that rail as it
      // was, so a rail showing no panel — emptied by the float, or collapsed — hands focus to the center instead.
      if (removed.restoresCenter) {
        focusRegion('center', typeId);
        return;
      }

      const isRailPanelShown =
        removed.returnRegion === 'left' ? isLeftPanelShown : removed.returnRegion === 'right' && isRightPanelShown;

      focusRegion(isRailPanelShown ? removed.returnRegion : 'center');
    },
    [focusRegion, isLeftPanelShown, isRightPanelShown, placementProject, widgets]
  );
  // The enable menu stays open and keeps focus, so removing a window from it moves none.
  const toggleRailItem = useCallback(
    (region: WidgetBarGroup['region'], item: (typeof leftMenuItems)[number]) =>
      item.isEnabled && item.isFloating
        ? removeFloatingPlacement({ instanceId: item.id, project: placementProject, widgets })
        : item.isEnabled
          ? closeWidgetPlacement({ widgets, getWidgetById, instanceId: item.id, project: placementProject, region })
          : openWidgetPlacement({
              widgets,
              getWidgetsForRegion,
              options: { createNew: item.allowMultiple, preferredRegions: [region] },
              typeId: item.typeId,
            }),
    [placementProject, widgets]
  );
  const handleToggleLeft = useCallback(
    (item: (typeof leftMenuItems)[number]) => toggleRailItem('left', item),
    [toggleRailItem]
  );
  const handleToggleRight = useCallback(
    (item: (typeof rightMenuItems)[number]) => toggleRailItem('right', item),
    [toggleRailItem]
  );
  const leftRailGroups = useMemo(
    () => [
      {
        activeId: panels.isLeftOpen && !leftRegion.isCollapsed ? leftRegion.activeInstanceId : null,
        dropState: leftDropState,
        railItems: leftRailItems,
        region: 'left' as const,
      },
    ],
    [leftDropState, leftRailItems, leftRegion.activeInstanceId, leftRegion.isCollapsed, panels.isLeftOpen]
  );
  const rightRailGroups = useMemo(
    () => [
      {
        activeId: panels.isRightOpen && !rightRegion.isCollapsed ? rightRegion.activeInstanceId : null,
        dropState: rightDropState,
        railItems: rightRailItems,
        region: 'right' as const,
      },
    ],
    [panels.isRightOpen, rightDropState, rightRailItems, rightRegion.activeInstanceId, rightRegion.isCollapsed]
  );

  return (
    <DndContext
      autoScroll={workbenchAutoScroll}
      collisionDetection={widgetCollisionDetection}
      modifiers={DND_MODIFIERS}
      sensors={sensors}
      onDragCancel={handleDragCancel}
      onDragEnd={handleDragEnd}
      onDragStart={handleDragStart}
    >
      <GalleryDragScope value>
        <Flex direction="column" h="100vh" w="100vw">
          <WorkbenchNotificationToaster />
          <DocumentTitleProgress />
          <TopBar />
          <ProjectConflictBanner />
          <QueueRecoveryBanner />

          <Flex aria-labelledby="workbench-project-heading" as="main" flex="1" minH="0" overflow="hidden">
            <VisuallyHidden as="h1" id="workbench-project-heading">
              {projectName}
            </VisuallyHidden>
            {/* Use a named content region: the project switcher is no longer a tablist. */}
            <Flex
              aria-labelledby="workbench-project-heading"
              flex="1"
              id={PROJECT_CONTENT_PANEL_ID}
              minH="0"
              overflow="hidden"
              role="region"
            >
              <WidgetBar
                edgeRegion={isLeftPanelShown ? 'left' : 'center'}
                groups={leftRailGroups}
                menuItems={leftMenuItems}
                projectId={projectId}
                side="left"
                onDock={handleDock}
                onRemoveFloating={handleRemoveFloating}
                onSelect={handleSelect}
                onToggle={handleToggleLeft}
              />
              {isLeftPanelShown ? <LeftPanel instanceId={leftRegion.activeInstanceId} /> : null}
              <CenterArea />
              {isRightPanelShown ? <RightPanel instanceId={rightRegion.activeInstanceId} /> : null}
              <WidgetBar
                edgeRegion={isRightPanelShown ? 'right' : 'center'}
                groups={rightRailGroups}
                menuItems={rightMenuItems}
                projectId={projectId}
                side="right"
                onDock={handleDock}
                onRemoveFloating={handleRemoveFloating}
                onSelect={handleSelect}
                onToggle={handleToggleRight}
              />
            </Flex>
          </Flex>

          <BottomPanel />
          <StatusBar dropState={bottomDropState} />
        </Flex>
        <FloatingWidgetLayer />
        <GalleryDragCursor />
        <PasteMediaRuntime />
        {/*
         * Allow pointer events through the full-size drag overlay so another finger can reach Preview pinch
         * handlers.
         */}
        <DragOverlay style={DRAG_OVERLAY_STYLE}>
          {activeDrag ? <WidgetDragPreview activeDrag={activeDrag} /> : null}
        </DragOverlay>
      </GalleryDragScope>
    </DndContext>
  );
};

const DRAG_OVERLAY_STYLE = { pointerEvents: 'none' } as const;

const WidgetDragPreview = ({ activeDrag }: { activeDrag: ActiveWidgetDrag }) => (
  <HStack bg="bg" borderWidth="1px" gap="2" px="3" py="2" rounded="md" shadow="lg">
    <WidgetIcon icon={activeDrag.icon} boxSize="4" />
    <Text fontSize="md" fontWeight="700">
      {activeDrag.label}
    </Text>
  </HStack>
);
