import { Flex } from '@chakra-ui/react';
import { useEffect } from 'react';

import { BottomPanel } from './components/BottomPanel';
import { CenterArea } from './components/CenterArea';
import { LeftPanel, RightPanel } from './components/Panels';
import { ShellErrorSurface } from './components/ShellErrorSurface';
import { StatusBar } from './components/StatusBar';
import { TopBar } from './components/TopBar';
import { WidgetBar, type WidgetBarItem } from './components/WidgetBar';
import { getWidgetsForRegion, widgetRegistrationFailures } from './widgetRegistry';
import { useWorkbench } from './WorkbenchContext';
import type { WidgetId, WidgetRegion } from './types';

const getWidgetBarItems = (region: WidgetRegion, enabledWidgetIds: WidgetId[]): WidgetBarItem[] =>
  getWidgetsForRegion(region).map((widget) => ({
    failureMessage: widget.failure?.message,
    id: widget.manifest.id,
    iconId: widget.manifest.icon,
    isEnabled: enabledWidgetIds.includes(widget.manifest.id),
    label: widget.manifest.labelText,
    status: widget.status,
  }));

/**
 * Top-level v7 workbench shell (Phase 1 skeleton).
 *
 * Composes the fixed Invoke control + project tabs, the left/right widget bars,
 * project-owned side panels, the center work area, the status bar, and the
 * copyable error surface. Layout regions are flex children that mount/unmount
 * from project layout state, which is more robust than the prototype's CSS grid
 * whose column template had to stay in lockstep with conditional children.
 */
export const WorkbenchShell = () => {
  const { activeProject, dispatch } = useWorkbench();
  const { panels } = activeProject.layout;
  const leftRegion = activeProject.widgetRegions.left;
  const rightRegion = activeProject.widgetRegions.right;
  const leftMenuItems = getWidgetBarItems('left', leftRegion.enabledWidgetIds);
  const rightMenuItems = getWidgetBarItems('right', rightRegion.enabledWidgetIds);
  const leftRailItems = leftMenuItems.filter((item) => item.isEnabled && item.status !== 'disabled');
  const rightRailItems = rightMenuItems.filter((item) => item.isEnabled && item.status !== 'disabled');
  const canShowLeftPanel = leftRailItems.some((item) => item.id === leftRegion.activeWidgetId);
  const canShowRightPanel = rightRailItems.some((item) => item.id === rightRegion.activeWidgetId);

  useEffect(() => {
    for (const failure of widgetRegistrationFailures) {
      dispatch({ failure, type: 'recordWidgetFailure' });
    }
  }, [dispatch]);

  return (
    <Flex direction="column" h="100vh" w="100vw">
      <TopBar />

      <Flex as="main" flex="1" minH="0" overflow="hidden">
        <WidgetBar
          activeId={panels.isLeftOpen && !leftRegion.isCollapsed ? leftRegion.activeWidgetId : null}
          menuItems={leftMenuItems}
          railItems={leftRailItems}
          region="left"
          side="left"
          onSelect={(widgetId) => dispatch({ region: 'left', type: 'selectRegionWidget', widgetId })}
          onToggle={(widgetId) => dispatch({ region: 'left', type: 'toggleRegionWidget', widgetId })}
        />
        {panels.isLeftOpen && !leftRegion.isCollapsed && canShowLeftPanel ? (
          <LeftPanel widgetId={leftRegion.activeWidgetId} />
        ) : null}
        <CenterArea />
        {panels.isRightOpen && !rightRegion.isCollapsed && canShowRightPanel ? (
          <RightPanel widgetId={rightRegion.activeWidgetId} />
        ) : null}
        <WidgetBar
          activeId={panels.isRightOpen && !rightRegion.isCollapsed ? rightRegion.activeWidgetId : null}
          menuItems={rightMenuItems}
          railItems={rightRailItems}
          region="right"
          side="right"
          onSelect={(widgetId) => dispatch({ region: 'right', type: 'selectRegionWidget', widgetId })}
          onToggle={(widgetId) => dispatch({ region: 'right', type: 'toggleRegionWidget', widgetId })}
        />
      </Flex>

      <BottomPanel />
      <StatusBar />
      <ShellErrorSurface />
    </Flex>
  );
};
