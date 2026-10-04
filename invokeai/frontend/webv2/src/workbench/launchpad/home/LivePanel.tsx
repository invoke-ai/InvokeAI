import { lazy, Suspense } from 'react';

/**
 * Load all live panels from one shared chunk while mounting independently. Use a null fallback because empty
 * panels render nothing.
 */

const loadLivePanels = () => import('./livePanels');

const ModelsNotice = lazy(() => loadLivePanels().then((module) => ({ default: module.ModelsNotice })));
const QueueStatusBand = lazy(() => loadLivePanels().then((module) => ({ default: module.QueueStatusBand })));
const RecentOutputs = lazy(() => loadLivePanels().then((module) => ({ default: module.RecentOutputs })));

const PANELS = {
  models: ModelsNotice,
  outputs: RecentOutputs,
  queue: QueueStatusBand,
} as const;

export const LivePanel = ({ panel }: { panel: keyof typeof PANELS }) => {
  const Panel = PANELS[panel];

  return (
    <Suspense fallback={null}>
      <Panel />
    </Suspense>
  );
};
