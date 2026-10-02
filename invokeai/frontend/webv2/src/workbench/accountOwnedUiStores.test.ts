import { accountLifecycle } from '@platform/state/accountLifecycle';
import { describe, expect, it } from 'vitest';

import type { ImageMapPoints } from './image-map/api';

import { isHotkeyModalLayerActive, registerHotkeyModalLayer } from './hotkeys/modalLayer';
import { imageMapStore } from './image-map/imageMapStore';
import { commandPaletteStore } from './palette/paletteStore';
import { openWorkbenchSettings, settingsDialogStore } from './settings/settingsDialogStore';
import { getLayerPropertiesRequest, requestLayerProperties } from './widgets/layers/layerPropertiesRequestStore';

describe('account-owned workbench UI stores', () => {
  it('synchronously removes transient UI state on account invalidation', () => {
    accountLifecycle.activate('user-a');
    commandPaletteStore.setSnapshot({ isOpen: true });
    openWorkbenchSettings('developer');
    const unregisterModal = registerHotkeyModalLayer('settings');
    requestLayerProperties('user-a-layer');
    // Partial stand-in: the snapshot only needs to be observably non-empty.
    imageMapStore.patchSnapshot({
      clusterLabels: { '0': { alternates: [], label: 'cats' } },
      data: { pointCount: 1, state: 'ready' } as unknown as ImageMapPoints,
      indexCounts: { embedded: 1, failed: 0, pending: 0, total: 1 },
      loadState: 'loaded',
    });

    accountLifecycle.invalidate();

    expect(commandPaletteStore.getSnapshot().isOpen).toBe(false);
    expect(settingsDialogStore.getSnapshot()).toMatchObject({
      isOpen: false,
      sectionId: 'appearance',
      query: '',
      returnFocus: null,
    });
    expect(settingsDialogStore.getSnapshot().target).toBeUndefined();
    expect(isHotkeyModalLayerActive()).toBe(false);
    expect(getLayerPropertiesRequest()).toBeNull();
    const imageMapSnapshot = imageMapStore.getSnapshot();
    expect(imageMapSnapshot.data).toBeNull();
    expect(imageMapSnapshot.loadState).toBe('idle');
    expect(imageMapSnapshot.error).toBeNull();
    expect(imageMapSnapshot.indexCounts).toBeNull();
    expect(imageMapSnapshot.clusterLabels).toBeNull();
    unregisterModal();
  });
});
