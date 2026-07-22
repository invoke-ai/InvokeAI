import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import { useMountEffect } from '@platform/react/useMountEffect';
import { formatHotkeyForPlatform } from '@workbench/hotkeys/keys';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { lazy, Suspense } from 'react';
import { tinykeys } from 'tinykeys';

import type { SettingsEntryDeps } from './entries';

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
        preferences={preferences as WorkbenchPreferences}
        settingsEntryDeps={SETTINGS_ENTRY_DEPS as SettingsEntryDeps}
        onClose={closeCommandPalette}
      />
    </Suspense>
  ) : null;
};
