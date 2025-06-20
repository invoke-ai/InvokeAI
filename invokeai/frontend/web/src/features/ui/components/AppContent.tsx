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
import { memo } from 'react';

import { TabMountGate } from './TabMountGate';
import ModelManagerTab from './tabs/ModelManagerTab';
import QueueTab from './tabs/QueueTab';

export const AppContent = memo(() => {
  const tabIndex = useAppSelector(selectActiveTabIndex);
  useDndMonitor();

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
