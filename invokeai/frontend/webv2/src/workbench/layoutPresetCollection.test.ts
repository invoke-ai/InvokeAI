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
  layoutPresetOrder: ['custom-1', 'compose', 'edit', 'video', 'automate'],
};

describe('layout preset collection', () => {
  it('drops stale and duplicate ids before slotting in missing presets', () => {
    expect(normalizeLayoutPresetOrder(['automate', 'bogus', 'automate'], [...layoutPresets, customPreset])).toEqual([
      'compose',
      'edit',
      'video',
      'automate',
      'custom-1',
    ]);
  });

  it('slots a newly shipped built-in beside its neighbour instead of appending it', () => {
    // Use pre-Video saved order to catch unconditional append behavior.
    expect(normalizeLayoutPresetOrder(['compose', 'edit', 'automate'], layoutPresets)).toEqual([
      'compose',
      'edit',
      'video',
      'automate',
    ]);
  });

  it('leaves a reordered strip reordered when it slots one in', () => {
    expect(
      normalizeLayoutPresetOrder(['custom-1', 'compose', 'edit', 'automate'], [...layoutPresets, customPreset])
    ).toEqual(['custom-1', 'compose', 'edit', 'video', 'automate']);
  });

  it('still appends a custom preset the order has never seen', () => {
    expect(
      normalizeLayoutPresetOrder(['compose', 'edit', 'video', 'automate'], [...layoutPresets, customPreset])
    ).toEqual(['compose', 'edit', 'video', 'automate', 'custom-1']);
  });

  it('resolves built-in and custom presets through the account order', () => {
    expect(getOrderedLayoutPresets(account).map(({ id }) => id)).toEqual([
      'custom-1',
      'compose',
      'edit',
      'video',
      'automate',
    ]);
  });

  it('moves a preset relative to the drop target without losing ids', () => {
    expect(reorderLayoutPresetIds(account, 'compose', 'automate')).toEqual([
      'custom-1',
      'edit',
      'video',
      'automate',
      'compose',
    ]);
  });

  it('returns null for invalid and unchanged moves', () => {
    expect(reorderLayoutPresetIds(account, 'missing', 'automate')).toBeNull();
    expect(reorderLayoutPresetIds(account, 'compose', 'compose')).toBeNull();
  });
});
