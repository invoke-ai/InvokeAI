import { IS_MAC_OS } from '@workbench/hotkeys/keys';
import { describe, expect, it } from 'vitest';

import { formatTopbarShortcut, formatTopbarShortcutForAria } from './useTopbarShortcut';

/**
 * The bar prints its own bindings on control faces, so it has to print the ones
 * this keyboard actually has. `⌘↵` on Linux names a key that is not there, which
 * is worse than showing nothing.
 */
describe('topbar shortcut labels', () => {
  it('renders modifiers for the running platform', () => {
    expect(formatTopbarShortcut('mod+enter')).toBe(IS_MAC_OS ? '⌘↵' : 'Ctrl+Enter');
    expect(formatTopbarShortcut('mod+s')).toBe(IS_MAC_OS ? '⌘S' : 'Ctrl+S');
    expect(formatTopbarShortcut('alt+mod+enter')).toBe(IS_MAC_OS ? '⌘⌥↵' : 'Ctrl+Alt+Enter');
  });

  it('formats the configured binding for aria-keyshortcuts', () => {
    expect(formatTopbarShortcutForAria('mod+enter')).toBe(IS_MAC_OS ? 'Meta+Enter' : 'Control+Enter');
    expect(formatTopbarShortcutForAria('alt+mod+s')).toBe(IS_MAC_OS ? 'Meta+Alt+S' : 'Control+Alt+S');
  });
});
