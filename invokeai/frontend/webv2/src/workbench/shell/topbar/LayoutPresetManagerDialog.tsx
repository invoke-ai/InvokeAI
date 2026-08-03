import { lazy, Suspense } from 'react';

import { layoutPresetManagerStore } from './layoutPresetManagerStore';

/**
 * Mounts the preset manager on demand. Managing presets is a rare, deliberate
 * trip, so its dialog — three list sections plus a rename and a confirm dialog —
 * has no business in the editor's initial payload.
 */
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
