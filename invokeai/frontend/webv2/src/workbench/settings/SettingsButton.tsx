import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { SettingsIcon } from 'lucide-react';
import { lazy, Suspense, useCallback } from 'react';

import { closeWorkbenchSettings, openWorkbenchSettings, settingsDialogStore } from './settingsDialogStore';

/**
 * The top bar's settings entry point, kept deliberately thin.
 *
 * The dialog body reaches into workbench commands, the persistence service, and
 * every settings section, so importing it eagerly made the Launchpad pay for
 * the whole editor composition — aggregate project state included — before the
 * user had opened a project. The trigger owns nothing but the store
 * subscription; the body arrives on first open.
 */
const LazySettingsDialog = lazy(() => import('./SettingsDialog'));

export const SettingsButton = () => {
  const isOpen = settingsDialogStore.useSelector((snapshot) => snapshot.isOpen);
  const handleOpen = useCallback(() => openWorkbenchSettings(), []);

  return (
    <>
      <Tooltip content="Settings">
        <IconButton aria-label="Settings" size="sm" variant="ghost" onClick={handleOpen}>
          <SettingsIcon />
        </IconButton>
      </Tooltip>
      {isOpen ? (
        <Suspense fallback={null}>
          <LazySettingsDialog isOpen onClose={closeWorkbenchSettings} />
        </Suspense>
      ) : null}
    </>
  );
};
