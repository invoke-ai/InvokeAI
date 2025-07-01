import 'dockview/dist/styles/dockview.css';
import 'features/ui/styles/dockview-theme-invoke.css';

import { TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useDndMonitor } from 'features/dnd/useDndMonitor';
import {
  selectWithCanvasTab,
  selectWithGenerateTab,
  selectWithModelsTab,
  selectWithQueueTab,
  selectWithUpscalingTab,
  selectWithWorkflowsTab,
} from 'features/system/store/configSlice';
import { VerticalNavBar } from 'features/ui/components/VerticalNavBar';
import { CanvasTabAutoLayout } from 'features/ui/layouts/canvas-tab-auto-layout';
import { GenerateTabAutoLayout } from 'features/ui/layouts/generate-tab-auto-layout';
import { UpscalingTabAutoLayout } from 'features/ui/layouts/upscaling-tab-auto-layout';
import { WorkflowsTabAutoLayout } from 'features/ui/layouts/workflows-tab-auto-layout';
import { selectActiveTabIndex } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

import ModelManagerTab from './tabs/ModelManagerTab';
import QueueTab from './tabs/QueueTab';

export const AppContent = memo(() => {
  useDndMonitor();
  const tabIndex = useAppSelector(selectActiveTabIndex);
  const withGenerateTab = useAppSelector(selectWithGenerateTab);
  const withCanvasTab = useAppSelector(selectWithCanvasTab);
  const withUpscalingTab = useAppSelector(selectWithUpscalingTab);
  const withWorkflowsTab = useAppSelector(selectWithWorkflowsTab);
  const withModelsTab = useAppSelector(selectWithModelsTab);
  const withQueueTab = useAppSelector(selectWithQueueTab);

  return (
    <Tabs index={tabIndex} display="flex" w="full" h="full" p={0} overflow="hidden">
      <VerticalNavBar />
      <TabPanels w="full" h="full" p={0}>
        {withGenerateTab && (
          <TabPanel w="full" h="full" p={0}>
            <GenerateTabAutoLayout />
          </TabPanel>
        )}
        {withCanvasTab && (
          <TabPanel w="full" h="full" p={0}>
            <CanvasTabAutoLayout />
          </TabPanel>
        )}
        {withUpscalingTab && (
          <TabPanel w="full" h="full" p={0}>
            <UpscalingTabAutoLayout />
          </TabPanel>
        )}
        {withWorkflowsTab && (
          <TabPanel w="full" h="full" p={0}>
            <WorkflowsTabAutoLayout />
          </TabPanel>
        )}
        {withModelsTab && (
          <TabPanel w="full" h="full" p={0}>
            <ModelManagerTab />
          </TabPanel>
        )}
        {withQueueTab && (
          <TabPanel w="full" h="full" p={0}>
            <QueueTab />
          </TabPanel>
        )}
      </TabPanels>
    </Tabs>
  );
});
AppContent.displayName = 'AppContent';
