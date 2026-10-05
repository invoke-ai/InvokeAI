import type { CanvasColorLabel } from '@workbench/canvas-engine/api';

/** Swatch hexes tuned to read on both themes; keys are the PSD layer-color vocabulary. */
const COLOR_LABEL_HEX: Record<CanvasColorLabel, string> = {
  blue: '#3e82f7',
  gray: '#8d8d8d',
  green: '#46a758',
  orange: '#ee7d2c',
  red: '#e5484d',
  violet: '#8e6cef',
  yellow: '#d9b40e',
};

export const colorLabelHex = (label: CanvasColorLabel): string => COLOR_LABEL_HEX[label];

/** The one label catalog; the leaf action registry and the group menu both render from it. */
export const COLOR_LABEL_ITEMS: readonly {
  readonly value: CanvasColorLabel;
  readonly hex: string;
  readonly labelKey: string;
  readonly defaultLabel: string;
}[] = (['red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray'] as const).map((value) => ({
  defaultLabel: value[0]!.toUpperCase() + value.slice(1),
  hex: COLOR_LABEL_HEX[value],
  labelKey: `widgets.layers.labels.${value}`,
  value,
}));
