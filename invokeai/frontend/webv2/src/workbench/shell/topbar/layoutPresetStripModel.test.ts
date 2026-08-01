import type { LayoutPreset } from '@workbench/layoutContracts';

import { layoutPresets } from '@workbench/layoutPresets';
import { describe, expect, it } from 'vitest';

import { getPresetAccessibleName, getTopbarPresetTabs } from './layoutPresetStripModel';

const customPreset = (id: string, label: string): LayoutPreset => ({
  ...layoutPresets[0],
  id,
  isBuiltIn: false,
  label,
});

describe('topbar preset strip model', () => {
  it('keeps every custom preset in the tab strip', () => {
    const customPresets = [customPreset('custom-1', 'One'), customPreset('custom-2', 'Two')];

    expect(getTopbarPresetTabs(customPresets)).toMatchObject([
      { id: 'compose' },
      { id: 'edit' },
      { id: 'automate' },
      { id: 'custom-1' },
      { id: 'custom-2' },
    ]);
  });

  it('keeps the visible label in the accessible name when visual text is hidden', () => {
    expect(getPresetAccessibleName(layoutPresets[0], false)).toBe('Compose');
    expect(getPresetAccessibleName(layoutPresets[0], true)).toBe('Compose, unsaved changes');
  });
});
