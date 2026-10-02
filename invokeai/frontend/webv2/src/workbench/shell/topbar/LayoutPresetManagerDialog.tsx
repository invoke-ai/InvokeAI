import { lazy, Suspense } from 'react';

import { layoutPresetManagerStore } from './layoutPresetManagerStore';

/** Lazy-load the infrequently used preset manager outside the editor's initial payload. */
const LazyLayoutPresetManagerDialogBody = lazy(() =>
  import('./LayoutPresetManagerDialogBody').then((module) => ({ default: module.LayoutPresetManagerDialogBody }))
);

export const LayoutPresetManagerDialog = () => {
  const isOpen = layoutPresetManagerStore.useSelector((snapshot) => snapshot.isOpen);

  return isOpen ? (
    <Suspense fallback={null}>
      <LazyLayoutPresetManagerDialogBody />
    </Suspense>
  ) : null;
};
