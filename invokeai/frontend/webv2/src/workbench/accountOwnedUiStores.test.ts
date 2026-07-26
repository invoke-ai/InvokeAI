import { accountLifecycle } from '@platform/state/accountLifecycle';
import { describe, expect, it } from 'vitest';

import { isHotkeyModalLayerActive, registerHotkeyModalLayer } from './hotkeys/modalLayer';
import { commandPaletteStore } from './palette/paletteStore';
import { settingsDialogStore } from './settings/settingsDialogStore';
import { getLayerPropertiesRequest, requestLayerProperties } from './widgets/layers/layerPropertiesRequestStore';

describe('account-owned workbench UI stores', () => {
  it('synchronously removes transient UI state on account invalidation', () => {
    accountLifecycle.activate('user-a');
    commandPaletteStore.setSnapshot({ isOpen: true });
    settingsDialogStore.setSnapshot({ isOpen: true, sectionId: 'developer' });
    const unregisterModal = registerHotkeyModalLayer('settings');
    requestLayerProperties('user-a-layer', 'filter');

    accountLifecycle.invalidate();

    expect(commandPaletteStore.getSnapshot().isOpen).toBe(false);
    expect(settingsDialogStore.getSnapshot()).toEqual({ isOpen: false, sectionId: 'appearance' });
    expect(isHotkeyModalLayerActive()).toBe(false);
    expect(getLayerPropertiesRequest()).toBeNull();
    unregisterModal();
  });
});
