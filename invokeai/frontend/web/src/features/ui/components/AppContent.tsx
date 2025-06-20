import 'dockview/dist/styles/dockview.css';
import 'features/ui/styles/dockview-theme-invoke.css';

import { TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useDndMonitor } from 'features/dnd/useDndMonitor';
import { VerticalNavBar } from 'features/ui/components/VerticalNavBar';
import { CanvasTabAutoLayout } from 'features/ui/layouts/canvas-tab-auto-layout';
import { GenerateTabAutoLayout } from 'features/ui/layouts/generate-tab-auto-layout';
import { UpscalingTabAutoLayout } from 'features/ui/layouts/upscaling-tab-auto-layout';
import { WorkflowsTabAutoLayout } from 'features/ui/layouts/workflows-tab-auto-layout';
import { selectActiveTabIndex } from 'features/ui/store/uiSelectors';
import { $isLeftPanelOpen, $isRightPanelOpen } from 'features/ui/store/uiSlice';
import type { CSSProperties } from 'react';
import { memo } from 'react';

import { TabMountGate } from './TabMountGate';
import ModelManagerTab from './tabs/ModelManagerTab';
import QueueTab from './tabs/QueueTab';

const panelStyles: CSSProperties = { position: 'relative', height: '100%', width: '100%', minWidth: 0 };

const onLeftPanelCollapse = (isCollapsed: boolean) => $isLeftPanelOpen.set(!isCollapsed);
const onRightPanelCollapse = (isCollapsed: boolean) => $isRightPanelOpen.set(!isCollapsed);

export const AppContent = memo(() => {
  // const tab = useAppSelector(selectActiveTab);
  const tabIndex = useAppSelector(selectActiveTabIndex);
  // const imperativePanelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  useDndMonitor();

  // const withLeftPanel = useAppSelector(selectWithLeftPanel);
  // const leftPanelUsePanelOptions = useMemo<UsePanelOptions>(
  //   () => ({
  //     id: 'left-panel',
  //     minSizePx: LEFT_PANEL_MIN_SIZE_PX,
  //     defaultSizePx: LEFT_PANEL_MIN_SIZE_PX,
  //     imperativePanelGroupRef,
  //     panelGroupDirection: 'horizontal',
  //     onCollapse: onLeftPanelCollapse,
  //   }),
  //   []
  // );
  // const leftPanel = usePanel(leftPanelUsePanelOptions);
  // useRegisteredHotkeys({
  //   id: 'toggleLeftPanel',
  //   category: 'app',
  //   callback: leftPanel.toggle,
  //   options: { enabled: withLeftPanel },
  //   dependencies: [leftPanel.toggle, withLeftPanel],
  // });

  // const withRightPanel = useAppSelector(selectWithRightPanel);
  // const rightPanelUsePanelOptions = useMemo<UsePanelOptions>(
  //   () => ({
  //     id: 'right-panel',
  //     minSizePx: RIGHT_PANEL_MIN_SIZE_PX,
  //     defaultSizePx: RIGHT_PANEL_MIN_SIZE_PX,
  //     imperativePanelGroupRef,
  //     panelGroupDirection: 'horizontal',
  //     onCollapse: onRightPanelCollapse,
  //   }),
  //   []
  // );
  // const rightPanel = usePanel(rightPanelUsePanelOptions);
  // useRegisteredHotkeys({
  //   id: 'toggleRightPanel',
  //   category: 'app',
  //   callback: rightPanel.toggle,
  //   options: { enabled: withRightPanel },
  //   dependencies: [rightPanel.toggle, withRightPanel],
  // });

  // useRegisteredHotkeys({
  //   id: 'resetPanelLayout',
  //   category: 'app',
  //   callback: () => {
  //     leftPanel.reset();
  //     rightPanel.reset();
  //   },
  //   dependencies: [leftPanel.reset, rightPanel.reset],
  // });
  // useRegisteredHotkeys({
  //   id: 'togglePanels',
  //   category: 'app',
  //   callback: () => {
  //     if (leftPanel.isCollapsed || rightPanel.isCollapsed) {
  //       leftPanel.expand();
  //       rightPanel.expand();
  //     } else {
  //       leftPanel.collapse();
  //       rightPanel.collapse();
  //     }
  //   },
  //   dependencies: [
  //     leftPanel.isCollapsed,
  //     rightPanel.isCollapsed,
  //     leftPanel.expand,
  //     rightPanel.expand,
  //     leftPanel.collapse,
  //     rightPanel.collapse,
  //   ],
  // });

  return (
    <Tabs index={tabIndex} display="flex" w="full" h="full" p={0} overflow="hidden">
      <TabList>
        <VerticalNavBar />
      </TabList>
      <TabPanels w="full" h="full" p={0}>
        <TabMountGate tab="generate">
          <TabPanel w="full" h="full" p={0}>
            <GenerateTabAutoLayout />
          </TabPanel>
        </TabMountGate>
        <TabMountGate tab="canvas">
          <TabPanel w="full" h="full" p={0}>
            <CanvasTabAutoLayout />
          </TabPanel>
        </TabMountGate>
        <TabMountGate tab="upscaling">
          <TabPanel w="full" h="full" p={0}>
            <UpscalingTabAutoLayout />
          </TabPanel>
        </TabMountGate>
        <TabMountGate tab="workflows">
          <TabPanel w="full" h="full" p={0}>
            <WorkflowsTabAutoLayout />
          </TabPanel>
        </TabMountGate>
        <TabMountGate tab="models">
          <TabPanel w="full" h="full" p={0}>
            <ModelManagerTab />
          </TabPanel>
        </TabMountGate>
        <TabMountGate tab="queue">
          <TabPanel w="full" h="full" p={0}>
            <QueueTab />
          </TabPanel>
        </TabMountGate>
      </TabPanels>
    </Tabs>
  );
});
AppContent.displayName = 'AppContent';
