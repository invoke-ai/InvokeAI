import { useAppSelector } from 'app/store/storeHooks';
import { AdvancedSession } from 'features/controlLayers/components/AdvancedSession/AdvancedSession';
import { SimpleSession } from 'features/controlLayers/components/SimpleSession/SimpleSession';
import { selectCanvasSessionId, selectGenerateSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import ModelManagerTab from 'features/ui/components/tabs/ModelManagerTab';
import QueueTab from 'features/ui/components/tabs/QueueTab';
import { WorkflowsMainPanel } from 'features/ui/components/tabs/WorkflowsTabContent';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { atom } from 'nanostores';
import { memo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

export const $simpleId = atom<string | null>(null);
export const $advancedId = atom<string | null>(null);

export const MainPanelContent = memo(() => {
  const tab = useAppSelector(selectActiveTab);
  const generateId = useAppSelector(selectGenerateSessionId);
  const canvasId = useAppSelector(selectCanvasSessionId);

  if (tab === 'generate') {
    return <SimpleSession id={generateId} />;
  }
  if (tab === 'canvas') {
    return <AdvancedSession id={canvasId} />;
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
