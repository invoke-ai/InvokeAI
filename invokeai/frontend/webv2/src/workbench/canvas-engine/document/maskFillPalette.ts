import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';

import { DEFAULT_INPAINT_MASK_FILL, DEFAULT_REGIONAL_GUIDANCE_FILL_COLOR } from './layerFactories';

// Kept out of canvas-engine/api.ts: the account registration is a module side effect, and the eager editor bundle
// reaches that barrel, so re-exporting it there would pull layer factories into editor startup.
const MASK_FILL_COLORS = [
  DEFAULT_INPAINT_MASK_FILL.color,
  DEFAULT_REGIONAL_GUIDANCE_FILL_COLOR,
  '#F59E42',
  '#F4D35E',
  '#38C9C5',
  '#62B5F5',
  '#637BE8',
  '#AD80E8',
  '#ED79BA',
  '#B7D65A',
] as const;

type MaskLayerKind = 'inpaint_mask' | 'regional_guidance';

const firstColorIndex = { inpaint_mask: 0, regional_guidance: 1 };
const nextColorIndex = { ...firstColorIndex };
registerAccountOwnedResource({
  clear: () => {
    Object.assign(nextColorIndex, firstColorIndex);
  },
  name: 'canvas-mask-fill-colors',
});

export const nextMaskFillColor = (kind: MaskLayerKind, offset = 0): string =>
  MASK_FILL_COLORS[(nextColorIndex[kind] + offset) % MASK_FILL_COLORS.length]!;

export const advanceMaskFillColors = (kind: MaskLayerKind, count = 1): void => {
  nextColorIndex[kind] = (nextColorIndex[kind] + count) % MASK_FILL_COLORS.length;
};
