import { useAppSelector } from 'app/store/storeHooks';
import { CanvasMainPanelContent } from 'features/controlLayers/components/CanvasMainPanelContent';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import ModelManagerTab from 'features/ui/components/tabs/ModelManagerTab';
import QueueTab from 'features/ui/components/tabs/QueueTab';
import { WorkflowsMainPanel } from 'features/ui/components/tabs/WorkflowsTabContent';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const MainPanelContent = memo(() => {
  const tab = useAppSelector(selectActiveTab);

  if (tab === 'canvas') {
    return <CanvasMainPanelContent />;
  }
  if (tab === 'upscaling') {
    return <ImageViewer />;
  }
  if (tab === 'workflows') {
    return <WorkflowsMainPanel />;
  }
  if (tab === 'models') {
    return <ModelManagerTab />;
  }
  if (tab === 'queue') {
    return <QueueTab />;
  }

  assert<Equals<never, typeof tab>>(false);
});
MainPanelContent.displayName = 'MainPanelContent';
