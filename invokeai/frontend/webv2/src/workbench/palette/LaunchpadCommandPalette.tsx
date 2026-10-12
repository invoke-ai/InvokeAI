import { useExitPresence } from '@platform/react/useExitRetainedValue';
import { useMountEffect } from '@platform/react/useMountEffect';
import { OPEN_COMMAND_PALETTE_HOTKEY } from '@workbench/hotkeys/catalog';
import { MOD_KEY_LABEL, toTinykeysBinding } from '@workbench/hotkeys/keys';
import { applyCustomHotkeys } from '@workbench/hotkeys/resolve';
import { useWorkbenchPreferences, useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { lazy, Suspense } from 'react';
import { tinykeys } from 'tinykeys';

import { closeCommandPalette, toggleCommandPalette, useIsCommandPaletteOpen } from './paletteStore';
import { SETTINGS_ENTRY_DEPS } from './settingsEntryDeps';

const LazyLaunchpadCommandPaletteDialog = lazy(() => import('./LaunchpadCommandPaletteDialog'));

const LaunchpadCommandPaletteHotkeys = ({ keys }: { keys: readonly string[] }) => {
  useMountEffect(() => {
    if (keys.length === 0) {
      return;
    }

    const onHotkey = (event: KeyboardEvent): void => {
      event.preventDefault();
      toggleCommandPalette();
    };
    const bindings = Object.fromEntries(keys.map((key) => [toTinykeysBinding(key), onHotkey]));

    return tinykeys(window, bindings, { ignore: () => false });
  });

  return null;
};

/** Lightweight Launchpad runtime and lazy dialog host; the dialog stays mounted while it animates closed. */
export const LaunchpadCommandPalette = () => {
  const isOpen = useIsCommandPaletteOpen();
  const dialog = useExitPresence(isOpen);
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const paletteHotkeys = applyCustomHotkeys(OPEN_COMMAND_PALETTE_HOTKEY, customHotkeys).keys;

  return (
    <>
      <LaunchpadCommandPaletteHotkeys key={paletteHotkeys.join('\n')} keys={paletteHotkeys} />
      {dialog.isMounted ? (
        <MountedLaunchpadCommandPalette key={dialog.generation} isOpen={isOpen} onExitComplete={dialog.release} />
      ) : null}
    </>
  );
};

const MountedLaunchpadCommandPalette = ({
  isOpen,
  onExitComplete,
}: {
  isOpen: boolean;
  onExitComplete: () => void;
}) => {
  const preferences = useWorkbenchPreferences();

  return (
    <Suspense fallback={null}>
      <LazyLaunchpadCommandPaletteDialog
        isOpen={isOpen}
        modifierKeyLabel={MOD_KEY_LABEL}
        preferences={preferences}
        settingsEntryDeps={SETTINGS_ENTRY_DEPS}
        onClose={closeCommandPalette}
        onExitComplete={onExitComplete}
      />
    </Suspense>
  );
};
