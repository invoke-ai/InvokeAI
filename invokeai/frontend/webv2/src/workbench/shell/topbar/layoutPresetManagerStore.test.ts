import { afterEach, describe, expect, it } from 'vitest';

import {
  closeLayoutPresetAdmin,
  layoutPresetManagerStore,
  openLayoutPresetDelete,
  openLayoutPresetEdit,
} from './layoutPresetManagerStore';

describe('layout preset administration store', () => {
  afterEach(closeLayoutPresetAdmin);

  it('opens one edit or delete target at a time and clears both on close', () => {
    openLayoutPresetEdit('custom-1');
    expect(layoutPresetManagerStore.getSnapshot()).toMatchObject({ deletePresetId: null, editPresetId: 'custom-1' });

    openLayoutPresetDelete('custom-2');
    expect(layoutPresetManagerStore.getSnapshot()).toMatchObject({ deletePresetId: 'custom-2', editPresetId: null });

    closeLayoutPresetAdmin();
    expect(layoutPresetManagerStore.getSnapshot()).toMatchObject({ deletePresetId: null, editPresetId: null });
  });
});
