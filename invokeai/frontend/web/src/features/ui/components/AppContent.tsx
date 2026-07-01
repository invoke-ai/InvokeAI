import 'dockview/dist/styles/dockview.css';
import 'features/ui/styles/dockview-theme-invoke.css';

import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import Loading from 'common/components/Loading/Loading';
import { useIsCustomNodesEnabled } from 'features/customNodes/useIsCustomNodesEnabled';
import { VerticalNavBar } from 'features/ui/components/VerticalNavBar';
import { CanvasTabAutoLayout } from 'features/ui/layouts/canvas-tab-auto-layout';
import { CustomNodesTabAutoLayout } from 'features/ui/layouts/customnodes-tab-auto-layout';
import { GenerateTabAutoLayout } from 'features/ui/layouts/generate-tab-auto-layout';
import { ModelsTabAutoLayout } from 'features/ui/layouts/models-tab-auto-layout';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { QueueTabAutoLayout } from 'features/ui/layouts/queue-tab-auto-layout';
import { UpscalingTabAutoLayout } from 'features/ui/layouts/upscaling-tab-auto-layout';
import { WorkflowsTabAutoLayout } from 'features/ui/layouts/workflows-tab-auto-layout';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useEffect } from 'react';

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
  const { isKnown: isCustomNodesKnown, isAllowed: isCustomNodesAllowed } = useIsCustomNodesEnabled();

  // Redirect away from customNodes only once we *know* the user is denied.
  // While setup status is still loading (isKnown=false), we do nothing —
  // the tab content is already suppressed (isAllowed=false), and we avoid
  // kicking a legitimate single-user session off a persisted tab before
  // the query resolves.
  useEffect(() => {
    if (tab === 'customNodes' && isCustomNodesKnown && !isCustomNodesAllowed) {
      navigationApi.switchToTab('generate');
    }
  }, [tab, isCustomNodesKnown, isCustomNodesAllowed]);

  return (
    <Flex position="relative" w="full" h="full" overflow="hidden">
      {tab === 'generate' && <GenerateTabAutoLayout />}
      {tab === 'canvas' && <CanvasTabAutoLayout />}
      {tab === 'upscaling' && <UpscalingTabAutoLayout />}
      {tab === 'workflows' && <WorkflowsTabAutoLayout />}
      {tab === 'models' && <ModelsTabAutoLayout />}
      {tab === 'customNodes' && isCustomNodesAllowed && <CustomNodesTabAutoLayout />}
      {tab === 'queue' && <QueueTabAutoLayout />}
      <SwitchingTabsLoader />
    </Flex>
  );
});
TabContent.displayName = 'TabContent';

const SwitchingTabsLoader = memo(() => {
  const isSwitchingTabs = useStore(navigationApi.$isLoading);

  if (isSwitchingTabs) {
    return <Loading />;
  }

  return null;
});
SwitchingTabsLoader.displayName = 'SwitchingTabsLoader';
