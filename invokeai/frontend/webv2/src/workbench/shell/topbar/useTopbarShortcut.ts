import type { HotkeyDefinition } from '@workbench/hotkeys/types';

import { firstPartyHotkeyCatalog } from '@workbench/hotkeys/catalog';
import { formatHotkeyAriaLabel, formatHotkeyForPlatform, formatHotkeyLabel } from '@workbench/hotkeys/keys';
import { applyCustomHotkeys } from '@workbench/hotkeys/resolve';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';

const findDefinition = (commandId: string): HotkeyDefinition | undefined =>
  firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === commandId);

/** Display effective bindings, including remaps; return null for unbound commands. */
export const useTopbarShortcut = (commandId: string): string | null => {
  const binding = useTopbarShortcutBinding(commandId);

  return binding?.display ?? null;
};

export interface TopbarShortcutBinding {
  aria: string;
  display: string;
  /** Canonical per-key tokens (`cmd`, `enter`, `k`, …) for glyph rendering. */
  parts: string[];
}

export const useTopbarShortcutBinding = (commandId: string): TopbarShortcutBinding | null => {
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const definition = findDefinition(commandId);
  const firstKey = definition ? applyCustomHotkeys(definition, customHotkeys).keys[0] : undefined;

  return firstKey
    ? {
        aria: formatHotkeyAriaLabel(firstKey),
        display: formatHotkeyLabel(firstKey),
        parts: formatHotkeyForPlatform(firstKey),
      }
    : null;
};
