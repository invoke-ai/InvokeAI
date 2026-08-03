import { lazy, Suspense } from 'react';

import { closeWorkbenchSettings, settingsDialogStore } from './settingsDialogStore';

/**
 * Mounts the settings dialog wherever it is placed, driven entirely by the
 * store. Every surface that can open settings — the app menu, a widget header
 * gear, a command — writes to the same store, so the dialog body must not be
 * owned by any one trigger.
 *
 * The body reaches into workbench commands, the persistence service, and every
 * settings section, so importing it eagerly made the Launchpad pay for the whole
 * editor composition — aggregate project state included — before the user had
 * opened a project. It arrives on first open.
 */
const LazySettingsDialog = lazy(() => import('./SettingsDialog'));

export const SettingsDialogHost = () => {
  const isOpen = settingsDialogStore.useSelector((snapshot) => snapshot.isOpen);

  return isOpen ? (
    <Suspense fallback={null}>
      <LazySettingsDialog isOpen onClose={closeWorkbenchSettings} />
    </Suspense>
  ) : null;
};
