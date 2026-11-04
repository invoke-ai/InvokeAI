import { useExitPresence } from '@platform/react/useExitRetainedValue';
import { Dialog } from '@platform/ui/Dialog';
import { lazy, Suspense } from 'react';

import { widgetsDialogStore } from './widgetsDialogStore';

/** The list loads on first open, outside the editor's initial payload. */
const LazyWidgetsDialogBody = lazy(() =>
  import('./WidgetsDialogBody').then((module) => ({ default: module.WidgetsDialogBody }))
);

// The dialog is modal from the press that opened it, not from when its module arrives.
const PENDING_DIALOG = <Dialog.Pending />;

/** Hosts the one Widgets dialog every opener (palette, app menu, the rails' and center's widget menus) writes to. */
export const WidgetsDialog = () => {
  const isOpen = widgetsDialogStore.useSelector((snapshot) => snapshot.isOpen);
  // Stays mounted through the close animation instead of vanishing with the open state.
  const dialog = useExitPresence(isOpen);

  return dialog.isMounted ? (
    <Suspense fallback={isOpen ? PENDING_DIALOG : null}>
      <LazyWidgetsDialogBody isOpen={isOpen} onExitComplete={dialog.release} />
    </Suspense>
  ) : null;
};
