import type {
  CanvasControlLayerContract,
  CanvasImageRef,
  CanvasInpaintMaskLayerContract,
  CanvasLayerBaseContract,
  CanvasLayerSourceContract,
  CanvasMaskFillContract,
  CanvasPixelSourceContract,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createDefaultControlAdapter } from '@workbench/controlAdapters';

export const DEFAULT_INPAINT_MASK_FILL = { color: '#e07575', style: 'diagonal' } as const;
const MASK_FILL_COLORS = [
  '#e07575',
  '#83d683',
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

export const DEFAULT_REGIONAL_GUIDANCE_FILL_COLOR = MASK_FILL_COLORS[1];

export const nextMaskFillColor = (kind: MaskLayerKind, offset = 0): string =>
  MASK_FILL_COLORS[(nextColorIndex[kind] + offset) % MASK_FILL_COLORS.length]!;

export const advanceMaskFillColors = (kind: MaskLayerKind, count = 1): void => {
  nextColorIndex[kind] = (nextColorIndex[kind] + count) % MASK_FILL_COLORS.length;
};

export const nextNumberedName = (prefix: string, existingNames: readonly string[]): string => {
  const expression = new RegExp(`^${prefix} (\\d+)$`);
  const used = new Set(
    existingNames
      .map((name) => Number(expression.exec(name.trim())?.[1]))
      .filter((number) => Number.isInteger(number) && number > 0)
  );
  let number = 1;
  while (used.has(number)) {
    number += 1;
  }
  return `${prefix} ${String(number)}`;
};

export const nextInpaintMaskName = (names: readonly string[]): string => nextNumberedName('Inpaint Mask', names);
export const nextControlLayerName = (names: readonly string[]): string => nextNumberedName('Control Layer', names);
export const nextRegionalGuidanceName = (names: readonly string[]): string =>
  nextNumberedName('Regional Guidance', names);
export interface CreateMaskLayerFromImageInput {
  image: CanvasImageRef;
  rect: Rect;
  id: string;
  name: string;
  fill: CanvasMaskFillContract;
}

export const createInpaintMaskFromImage = (input: CreateMaskLayerFromImageInput): CanvasInpaintMaskLayerContract => ({
  blendMode: 'normal',
  id: input.id,
  isEnabled: true,
  isLocked: false,
  mask: {
    bitmap: { ...input.image },
    fill: { ...input.fill },
    offset: { x: input.rect.x, y: input.rect.y },
  },
  name: input.name,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

export const createRegionalGuidanceFromImage = (
  input: CreateMaskLayerFromImageInput
): CanvasRegionalGuidanceLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id: input.id,
  isEnabled: true,
  isLocked: false,
  mask: {
    bitmap: { ...input.image },
    fill: { ...input.fill },
    offset: { x: input.rect.x, y: input.rect.y },
  },
  name: input.name,
  negativePrompt: null,
  opacity: 0.5,
  positivePrompt: null,
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
});

export interface CreateControlLayerInput {
  name: string;
  id: string;
  /** Main model base whose adapter kind and defaults the layer starts on. */
  base?: string | null;
  /** Defaults to an empty paint source. */
  source?: CanvasPixelSourceContract;
}

/** The one control-layer factory: untyped, with no model, so generation resolves the model from the type. */
export const createControlLayer = ({
  base,
  id,
  name,
  source,
}: CreateControlLayerInput): CanvasControlLayerContract => ({
  adapter: createDefaultControlAdapter(base, null),
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name,
  opacity: 1,
  source: source ?? { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

/**
 * An untyped, model-less control layer made from another layer, keeping its appearance and flags. `content` either
 * places baked pixels at `rect` under an identity transform, or reuses a source and transform as they are.
 */
export const createControlLayerFrom = (
  from: CanvasLayerBaseContract,
  options: {
    id: string;
    name: string;
    base: string | null;
    content: { readonly rect: Rect } | { readonly source: CanvasLayerSourceContract };
  }
): CanvasControlLayerContract => ({
  ...createControlLayer({ base: options.base, id: options.id, name: options.name }),
  blendMode: from.blendMode,
  ...(from.colorLabel ? { colorLabel: from.colorLabel } : {}),
  isEnabled: from.isEnabled,
  isLocked: from.isLocked,
  opacity: from.opacity,
  ...('rect' in options.content
    ? { source: { bitmap: null, offset: { x: options.content.rect.x, y: options.content.rect.y }, type: 'paint' } }
    : { source: structuredClone(options.content.source), transform: { ...from.transform } }),
});
