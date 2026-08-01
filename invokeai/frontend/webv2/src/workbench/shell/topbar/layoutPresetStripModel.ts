import type { LayoutPreset } from '@workbench/layoutContracts';

import { layoutPresets } from '@workbench/layoutPresets';

export const getTopbarPresetTabs = (customPresets: LayoutPreset[]): LayoutPreset[] => [
  ...layoutPresets,
  ...customPresets,
];

export const getPresetAccessibleName = (
  preset: LayoutPreset,
  hasUnsavedChanges: boolean,
  unsavedChangesLabel = 'unsaved changes'
): string => (hasUnsavedChanges ? `${preset.label}, ${unsavedChangesLabel}` : preset.label);
