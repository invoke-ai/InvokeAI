import { describe, expect, it } from 'vitest';

import type { LayoutPreset } from './layoutContracts';
import type { AccountState } from './projectContracts';

import { getOrderedLayoutPresets, normalizeLayoutPresetOrder, reorderLayoutPresetIds } from './layoutPresetCollection';
import { layoutPresets } from './layoutPresets';

const customPreset: LayoutPreset = {
  id: 'custom-1',
  label: 'Custom',
  snapshot: layoutPresets[0].snapshot,
};

const account: AccountState = {
  activeLayoutPresetId: 'compose',
  customLayoutPresets: [customPreset],
  layoutPresetOrder: ['custom-1', 'compose', 'edit', 'automate'],
};

describe('layout preset collection', () => {
  it('drops stale and duplicate ids before appending missing presets', () => {
    expect(normalizeLayoutPresetOrder(['automate', 'bogus', 'automate'], [...layoutPresets, customPreset])).toEqual([
      'automate',
      'compose',
      'edit',
      'custom-1',
    ]);
  });

  it('resolves built-in and custom presets through the account order', () => {
    expect(getOrderedLayoutPresets(account).map(({ id }) => id)).toEqual(['custom-1', 'compose', 'edit', 'automate']);
  });

  it('moves a preset relative to the drop target without losing ids', () => {
    expect(reorderLayoutPresetIds(account, 'compose', 'automate')).toEqual(['custom-1', 'edit', 'automate', 'compose']);
  });

  it('returns null for invalid and unchanged moves', () => {
    expect(reorderLayoutPresetIds(account, 'missing', 'automate')).toBeNull();
    expect(reorderLayoutPresetIds(account, 'compose', 'compose')).toBeNull();
  });
});
