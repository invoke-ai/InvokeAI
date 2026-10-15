import { requestQueueItemReveal } from '@features/queue/reveal';
import { useExitPresence } from '@platform/react/useExitRetainedValue';
import { Dialog } from '@platform/ui/Dialog';
import { firstPartyHotkeyCatalog } from '@workbench/hotkeys/catalog';
import { formatHotkeyForPlatform, MOD_KEY_LABEL } from '@workbench/hotkeys/keys';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { openWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { lazy, Suspense } from 'react';

import { closeCommandPalette, useIsCommandPaletteOpen } from './paletteStore';
import { SETTINGS_ENTRY_DEPS } from './settingsEntryDeps';

export const loadWorkbenchCommandPaletteDialog = () => import('./WorkbenchCommandPaletteDialog');

const LazyWorkbenchCommandPaletteDialog = lazy(loadWorkbenchCommandPaletteDialog);
// The palette is open, and modal, from the keypress that opened it, not from when its module arrives.
const PENDING_DIALOG = <Dialog.Pending />;

/** Lightweight route host; the palette implementation is loaded only while open or animating closed. */
export const WorkbenchCommandPalette = () => {
  const isOpen = useIsCommandPaletteOpen();
  const dialog = useExitPresence(isOpen);

  return dialog.isMounted ? (
    <MountedWorkbenchCommandPalette key={dialog.generation} isOpen={isOpen} onExitComplete={dialog.release} />
  ) : null;
};

const MountedWorkbenchCommandPalette = ({
  isOpen,
  onExitComplete,
}: {
  isOpen: boolean;
  onExitComplete: () => void;
}) => {
  const preferences = useWorkbenchPreferences();

  return (
    <Suspense fallback={isOpen ? PENDING_DIALOG : null}>
      <LazyWorkbenchCommandPaletteDialog
        catalog={firstPartyHotkeyCatalog}
        formatHotkey={formatHotkeyForPlatform}
        getWidgetsForRegion={getWidgetsForRegion}
        isOpen={isOpen}
        modifierKeyLabel={MOD_KEY_LABEL}
        openWidgetPlacement={openWidgetPlacement}
        preferences={preferences}
        requestQueueItemReveal={requestQueueItemReveal}
        settingsEntryDeps={SETTINGS_ENTRY_DEPS}
        onClose={closeCommandPalette}
        onExitComplete={onExitComplete}
      />
    </Suspense>
  );
};
