import { accountLifecycle } from '@platform/state/accountLifecycle';
import { describe, expect, it } from 'vitest';

import type { ImageMapPoints } from './image-map/api';

import { isHotkeyModalLayerActive, registerHotkeyModalLayer } from './hotkeys/modalLayer';
import { imageMapStore } from './image-map/imageMapStore';
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
    // Partial stand-in: the snapshot only needs to be observably non-empty.
    imageMapStore.patchSnapshot({
      data: { pointCount: 1, state: 'ready' } as unknown as ImageMapPoints,
      indexCounts: { embedded: 1, failed: 0, pending: 0, total: 1 },
      loadState: 'loaded',
    });

    accountLifecycle.invalidate();

    expect(commandPaletteStore.getSnapshot().isOpen).toBe(false);
    expect(settingsDialogStore.getSnapshot()).toEqual({ isOpen: false, sectionId: 'appearance' });
    expect(isHotkeyModalLayerActive()).toBe(false);
    expect(getLayerPropertiesRequest()).toBeNull();
    const imageMapSnapshot = imageMapStore.getSnapshot();
    expect(imageMapSnapshot.data).toBeNull();
    expect(imageMapSnapshot.loadState).toBe('idle');
    expect(imageMapSnapshot.error).toBeNull();
    expect(imageMapSnapshot.indexCounts).toBeNull();
    unregisterModal();
  });
});
