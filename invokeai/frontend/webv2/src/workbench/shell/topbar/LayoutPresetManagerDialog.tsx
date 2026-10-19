import { useExitPresence } from '@platform/react/useExitRetainedValue';
import { lazy, Suspense } from 'react';

import { layoutPresetManagerStore } from './layoutPresetManagerStore';

/** Lazy-load the infrequently used preset manager outside the editor's initial payload. */
const LazyLayoutPresetManagerDialogBody = lazy(() =>
  import('./LayoutPresetManagerDialogBody').then((module) => ({ default: module.LayoutPresetManagerDialogBody }))
);

export const LayoutPresetManagerDialog = () => {
  const isOpen = layoutPresetManagerStore.useSelector((snapshot) => snapshot.isOpen);
  // Stays mounted through the close animation instead of vanishing with the open state.
  const dialog = useExitPresence(isOpen);

  return dialog.isMounted ? (
    <Suspense fallback={null}>
      <LazyLayoutPresetManagerDialogBody isOpen={isOpen} onExitComplete={dialog.release} />
    </Suspense>
  ) : null;
};
