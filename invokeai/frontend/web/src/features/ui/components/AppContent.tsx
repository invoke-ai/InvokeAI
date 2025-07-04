import 'dockview/dist/styles/dockview.css';
import 'features/ui/styles/dockview-theme-invoke.css';

import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import Loading from 'common/components/Loading/Loading';
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
import { ModelsTabAutoLayout } from 'features/ui/layouts/models-tab-auto-layout';
import { QueueTabAutoLayout } from 'features/ui/layouts/queue-tab-auto-layout';
import { UpscalingTabAutoLayout } from 'features/ui/layouts/upscaling-tab-auto-layout';
import { WorkflowsTabAutoLayout } from 'features/ui/layouts/workflows-tab-auto-layout';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useState } from 'react';

export const AppContent = memo(() => {
  useDndMonitor();
  const tab = useAppSelector(selectActiveTab);
  const [isLoading, setIsLoading] = useState(true);
  const withGenerateTab = useAppSelector(selectWithGenerateTab);
  const withCanvasTab = useAppSelector(selectWithCanvasTab);
  const withUpscalingTab = useAppSelector(selectWithUpscalingTab);
  const withWorkflowsTab = useAppSelector(selectWithWorkflowsTab);
  const withModelsTab = useAppSelector(selectWithModelsTab);
  const withQueueTab = useAppSelector(selectWithQueueTab);

  return (
    <Flex position="relative" w="full" h="full" overflow="hidden">
      <VerticalNavBar />
      <Flex position="relative" w="full" h="full" overflow="hidden">
        {withGenerateTab && tab === 'generate' && <GenerateTabAutoLayout setIsLoading={setIsLoading} />}
        {withCanvasTab && tab === 'canvas' && <CanvasTabAutoLayout setIsLoading={setIsLoading} />}
        {withUpscalingTab && tab === 'upscaling' && <UpscalingTabAutoLayout setIsLoading={setIsLoading} />}
        {withWorkflowsTab && tab === 'workflows' && <WorkflowsTabAutoLayout setIsLoading={setIsLoading} />}
        {withModelsTab && tab === 'models' && <ModelsTabAutoLayout setIsLoading={setIsLoading} />}
        {withQueueTab && tab === 'queue' && <QueueTabAutoLayout setIsLoading={setIsLoading} />}
        {isLoading && <Loading />}
      </Flex>
    </Flex>
  );
});
AppContent.displayName = 'AppContent';
