import { useMountEffect } from '@platform/react/useMountEffect';
import { formatHotkeyForPlatform } from '@workbench/hotkeys/keys';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { lazy, Suspense } from 'react';
import { tinykeys } from 'tinykeys';

import { closeCommandPalette, toggleCommandPalette, useIsCommandPaletteOpen } from './paletteStore';
import { SETTINGS_ENTRY_DEPS } from './settingsEntryDeps';

const LazyLaunchpadCommandPaletteDialog = lazy(() => import('./LaunchpadCommandPaletteDialog'));

/** Lightweight Launchpad runtime and lazy dialog host. */
export const LaunchpadCommandPalette = () => {
  const isOpen = useIsCommandPaletteOpen();
  const preferences = useWorkbenchPreferences();

  useMountEffect(() =>
    tinykeys(window, {
      '$mod+KeyK': (event) => {
        event.preventDefault();
        toggleCommandPalette();
      },
    })
  );

  return isOpen ? (
    <Suspense fallback={null}>
      <LazyLaunchpadCommandPaletteDialog
        modifierKeyLabel={formatHotkeyForPlatform('mod')[0]!}
        preferences={preferences}
        settingsEntryDeps={SETTINGS_ENTRY_DEPS}
        onClose={closeCommandPalette}
      />
    </Suspense>
  ) : null;
};
