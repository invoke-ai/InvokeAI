import { lazy, Suspense } from 'react';

/**
 * Places one live panel on Home.
 *
 * All three resolve from the same `import('./livePanels')` promise, so the
 * bundler emits a single dynamic chunk for the lot rather than one per panel —
 * but each still mounts wherever the page wants it. Every panel renders
 * nothing until it has something to report, so the fallback is `null`: a
 * skeleton for a band that will usually turn out to be empty is worse than
 * nothing appearing at all.
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
