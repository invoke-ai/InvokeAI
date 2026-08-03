import type { LayoutPreset } from '@workbench/layoutContracts';
import type { LucideIcon } from 'lucide-react';

import { builtInLayoutPresetDescriptors } from '@workbench/layoutPresets';

import { resolveLayoutPresetIcon } from './layoutPresetIcons';

/**
 * Icon and tooltip for a preset. Deliberately not on `LayoutPreset` itself: that
 * type is persisted verbatim for custom presets, and a component reference has
 * no place in a serialized snapshot — which is why custom presets store an icon
 * *id* and resolve it here.
 *
 * The built-in tooltips say what the arrangement is *for*, since the labels name
 * a mode of working rather than a widget — "Automate" has to earn its name
 * somewhere.
 */
export interface LayoutPresetPresentation {
  icon: LucideIcon;
  tooltip: string;
}

const builtInPresentation = new Map(
  builtInLayoutPresetDescriptors.map(({ iconId, preset, tooltip }) => [
    preset.id,
    { icon: resolveLayoutPresetIcon(iconId), tooltip },
  ])
);

export const getLayoutPresetPresentation = (preset: LayoutPreset): LayoutPresetPresentation =>
  builtInPresentation.get(preset.id) ?? {
    icon: resolveLayoutPresetIcon(preset.iconId),
    tooltip: 'Custom layout',
  };
