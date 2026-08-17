import { Flex, HStack, Text, VisuallyHidden } from '@chakra-ui/react';
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
import { useMountEffect } from '@platform/react/useMountEffect';
import { FocusRegionProvider } from '@workbench/focusRegions';
import { WidgetIcon } from '@workbench/iconResolver';
import { PROJECT_CONTENT_PANEL_ID } from '@workbench/projects/projectTabsA11y';
import { WidgetBar } from '@workbench/widget-frame';
import { FloatingWidgetLayer } from '@workbench/widget-frame/FloatingWidgetLayer';
import {
  getRegionDropState,
  isWidgetDndData,
  isWidgetInstanceDragData,
  resolveWidgetDragEnd,
  type ActiveWidgetDrag,
  widgetCollisionDetection,
} from '@workbench/widgetDnd';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import {
  closeWidgetPlacement,
  dispatchWidgetDragEndPlacement,
  openWidgetPlacement,
  revealWidgetPlacement,
} from '@workbench/widgetPlacementCommands';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { createWidgetRegionViewModelFromState, getWidgetRegionItems } from '@workbench/widgetRegionViewModel';
import { getWidgetById, getWidgetsForRegion, widgetRegistrationFailures } from '@workbench/widgetRegistry';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { BottomPanel } from './BottomPanel';
import { CenterArea } from './CenterArea';
import { DocumentTitleProgress } from './DocumentTitleProgress';
import { WorkbenchNotificationToaster } from './notifications';
import { LeftPanel, RightPanel } from './Panels';
import { StatusBar } from './StatusBar';
import { TopBar } from './topbar';

const DND_MODIFIERS = [restrictToWindowEdges];

/**
 * Default 20% edge zones, at half the default scroll speed. The zones must
 * stay full-size: in the side layout the board list is under 250px tall, so
 * a narrower band excludes the first and last visible rows — the places an
 * image is actually held to scroll the list. Speed is halved because at
 * dnd-kit's default a list this small dumps its full range in under half a
 * second, far too fast to pick a row. Phantom off-screen targets triggering
 * scrolls from afar are prevented by the visibility check inside
 * widgetCollisionDetection, not by shrinking zones.
 */
const DND_AUTO_SCROLL = { acceleration: 5 };

export const WorkbenchShell = () => {
  const { notifications, widgets } = useWorkbenchCommands();
  const { t } = useTranslation();
  const panels = useActiveProjectSelector((project) => project.layout.panels);
  const projectName = useActiveProjectSelector((project) => project.name);
  const leftRegion = useActiveProjectSelector((project) => project.widgetRegions.left);
  const rightRegion = useActiveProjectSelector((project) => project.widgetRegions.right);
  const placementProject = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
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
        region: 'left',
        regionState: leftRegion,
        widgetInstances: placementProject.widgetInstances,
        widgets: getWidgetsForRegion('left'),
        getWidgetLabel,
      }),
    [getWidgetLabel, leftRegion, placementProject.widgetInstances]
  );
  const rightRegionViewModel = useMemo(
    () =>
      createWidgetRegionViewModelFromState({
        region: 'right',
        regionState: rightRegion,
        widgetInstances: placementProject.widgetInstances,
        widgets: getWidgetsForRegion('right'),
        getWidgetLabel,
      }),
    [getWidgetLabel, placementProject.widgetInstances, rightRegion]
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
  const handleSelectLeft = useCallback(
    (instanceId: string) => revealWidgetPlacement({ instanceId, project: placementProject, region: 'left', widgets }),
    [placementProject, widgets]
  );
  const handleSelectRight = useCallback(
    (instanceId: string) => revealWidgetPlacement({ instanceId, project: placementProject, region: 'right', widgets }),
    [placementProject, widgets]
  );
  const handleToggleLeft = useCallback(
    (item: (typeof leftMenuItems)[number]) =>
      item.isEnabled
        ? closeWidgetPlacement({
            widgets,
            getWidgetById,
            instanceId: item.id,
            project: placementProject,
            region: 'left',
          })
        : openWidgetPlacement({
            widgets,
            getWidgetsForRegion,
            options: { createNew: item.allowMultiple, preferredRegions: ['left'] },
            typeId: item.typeId,
          }),
    [placementProject, widgets]
  );
  const handleToggleRight = useCallback(
    (item: (typeof rightMenuItems)[number]) =>
      item.isEnabled
        ? closeWidgetPlacement({
            widgets,
            getWidgetById,
            instanceId: item.id,
            project: placementProject,
            region: 'right',
          })
        : openWidgetPlacement({
            widgets,
            getWidgetsForRegion,
            options: { createNew: item.allowMultiple, preferredRegions: ['right'] },
            typeId: item.typeId,
          }),
    [placementProject, widgets]
  );

  return (
    <FocusRegionProvider>
      <DndContext
        autoScroll={DND_AUTO_SCROLL}
        collisionDetection={widgetCollisionDetection}
        modifiers={DND_MODIFIERS}
        sensors={sensors}
        onDragCancel={handleDragCancel}
        onDragEnd={handleDragEnd}
        onDragStart={handleDragStart}
      >
        <Flex direction="column" h="100vh" w="100vw">
          <WorkbenchNotificationToaster />
          <DocumentTitleProgress />
          <TopBar />

          <Flex aria-labelledby="workbench-project-heading" as="main" flex="1" minH="0" overflow="hidden">
            <VisuallyHidden as="h1" id="workbench-project-heading">
              {projectName}
            </VisuallyHidden>
            {/* Not a tab panel any more: the project tab strip became a
                dropdown, so there is no tab list for this to belong to. It is
                the project's content region, named by the heading above it. */}
            <Flex
              aria-labelledby="workbench-project-heading"
              flex="1"
              id={PROJECT_CONTENT_PANEL_ID}
              minH="0"
              overflow="hidden"
              role="region"
            >
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
              <CenterArea />
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
          </Flex>

          <BottomPanel />
          <StatusBar dropState={bottomDropState} />
        </Flex>
        <FloatingWidgetLayer />
        <DragOverlay>{activeDrag ? <WidgetDragPreview activeDrag={activeDrag} /> : null}</DragOverlay>
      </DndContext>
    </FocusRegionProvider>
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
