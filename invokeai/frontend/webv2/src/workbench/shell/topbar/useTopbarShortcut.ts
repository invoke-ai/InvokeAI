import type { HotkeyDefinition } from '@workbench/hotkeys/types';

import { firstPartyHotkeyCatalog } from '@workbench/hotkeys/catalog';
import { formatHotkeyForPlatform, IS_MAC_OS } from '@workbench/hotkeys/keys';
import { applyCustomHotkeys } from '@workbench/hotkeys/resolve';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';

/**
 * Glyphs for the compact hints printed on control faces and menu rows.
 *
 * macOS spells modifiers as symbols and runs them together; everywhere else
 * they are words joined by `+`. Printing `⌘↵` on Linux would name a key that
 * keyboard does not have, which is worse than printing nothing.
 */
const MAC_GLYPHS: Record<string, string> = {
  alt: '⌥',
  cmd: '⌘',
  ctrl: '⌃',
  enter: '↵',
  option: '⌥',
  shift: '⇧',
};

const OTHER_LABELS: Record<string, string> = {
  alt: 'Alt',
  ctrl: 'Ctrl',
  enter: 'Enter',
  meta: 'Win',
  shift: 'Shift',
};

const formatPart = (part: string): string =>
  IS_MAC_OS ? (MAC_GLYPHS[part] ?? part.toUpperCase()) : (OTHER_LABELS[part] ?? part.toUpperCase());

/** One key's text label, for render sites that draw icons for some keys and need the text for the rest. */
export const formatTopbarShortcutPart = formatPart;

export const formatTopbarShortcut = (hotkey: string): string =>
  formatHotkeyForPlatform(hotkey)
    .map(formatPart)
    .join(IS_MAC_OS ? '' : '+');

const ARIA_LABELS: Record<string, string> = {
  alt: 'Alt',
  cmd: 'Meta',
  ctrl: 'Control',
  enter: 'Enter',
  option: 'Alt',
  shift: 'Shift',
};

export const formatTopbarShortcutForAria = (hotkey: string): string =>
  formatHotkeyForPlatform(hotkey)
    .map((part) => ARIA_LABELS[part] ?? (part.length === 1 ? part.toUpperCase() : part))
    .join('+');

const findDefinition = (commandId: string): HotkeyDefinition | undefined =>
  firstPartyHotkeyCatalog.find((hotkey) => hotkey.id === commandId);

/**
 * The binding a command is *actually* on, formatted for this platform, or null
 * when the user has unbound it. Reading the effective binding rather than the
 * default means a remapped Invoke does not advertise a key that no longer works.
 */
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
        aria: formatTopbarShortcutForAria(firstKey),
        display: formatTopbarShortcut(firstKey),
        parts: formatHotkeyForPlatform(firstKey),
      }
    : null;
};
