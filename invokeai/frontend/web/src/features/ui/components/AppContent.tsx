import 'dockview/dist/styles/dockview.css';
import 'features/ui/styles/dockview-theme-invoke.css';

import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import Loading from 'common/components/Loading/Loading';
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
import { ModelsTabAutoLayout } from 'features/ui/layouts/models-tab-auto-layout';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { QueueTabAutoLayout } from 'features/ui/layouts/queue-tab-auto-layout';
import { UpscalingTabAutoLayout } from 'features/ui/layouts/upscaling-tab-auto-layout';
import { WorkflowsTabAutoLayout } from 'features/ui/layouts/workflows-tab-auto-layout';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

export const AppContent = memo(() => {
  return (
    <Flex position="relative" w="full" h="full" overflow="hidden">
      <VerticalNavBar />
      <TabContent />
    </Flex>
  );
});
AppContent.displayName = 'AppContent';

const TabContent = memo(() => {
  const tab = useAppSelector(selectActiveTab);
  const withGenerateTab = useAppSelector(selectWithGenerateTab);
  const withCanvasTab = useAppSelector(selectWithCanvasTab);
  const withUpscalingTab = useAppSelector(selectWithUpscalingTab);
  const withWorkflowsTab = useAppSelector(selectWithWorkflowsTab);
  const withModelsTab = useAppSelector(selectWithModelsTab);
  const withQueueTab = useAppSelector(selectWithQueueTab);

  return (
    <Flex position="relative" w="full" h="full" overflow="hidden">
      {withGenerateTab && tab === 'generate' && <GenerateTabAutoLayout />}
      {withCanvasTab && tab === 'canvas' && <CanvasTabAutoLayout />}
      {withUpscalingTab && tab === 'upscaling' && <UpscalingTabAutoLayout />}
      {withWorkflowsTab && tab === 'workflows' && <WorkflowsTabAutoLayout />}
      {withModelsTab && tab === 'models' && <ModelsTabAutoLayout />}
      {withQueueTab && tab === 'queue' && <QueueTabAutoLayout />}
      <SwitchingTabsLoader />
    </Flex>
  );
});
TabContent.displayName = 'TabContent';

const SwitchingTabsLoader = memo(() => {
  const isSwitchingTabs = useStore(navigationApi.$isSwitchingTabs);

  if (isSwitchingTabs) {
    return <Loading />;
  }

  return null;
});
SwitchingTabsLoader.displayName = 'SwitchingTabsLoader';
