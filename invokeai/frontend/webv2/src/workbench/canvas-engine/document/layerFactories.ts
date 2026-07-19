import type {
  CanvasControlLayerContract,
  CanvasImageRef,
  CanvasInpaintMaskLayerContract,
  CanvasMaskFillContract,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { CONTROL_ADAPTER_DEFAULTS } from '@workbench/controlAdapters';

export const DEFAULT_INPAINT_MASK_FILL = { color: '#e07575', style: 'diagonal' } as const;
export const REGIONAL_GUIDANCE_FILL_COLORS: readonly string[] = [
  '#799ddb',
  '#83d683',
  '#fae150',
  '#dc9065',
  '#e07575',
  '#d58bca',
  '#a178d6',
];

const nextNumberedName = (prefix: string, existingNames: readonly string[]): string => {
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
export const nextRegionalGuidanceFillColor = (count: number): string =>
  REGIONAL_GUIDANCE_FILL_COLORS[(count + 1) % REGIONAL_GUIDANCE_FILL_COLORS.length] ??
  REGIONAL_GUIDANCE_FILL_COLORS[0]!;

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

export const createControlLayer = (name: string, id: string, base?: string | null): CanvasControlLayerContract => {
  const adapter = base === 'z-image' ? CONTROL_ADAPTER_DEFAULTS.z_image_control : CONTROL_ADAPTER_DEFAULTS.controlnet;
  return {
    adapter: { ...adapter, beginEndStepPct: [...adapter.beginEndStepPct] },
    blendMode: 'normal',
    id,
    isEnabled: true,
    isLocked: false,
    name,
    opacity: 1,
    source: { bitmap: null, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'control',
    withTransparencyEffect: true,
  };
};
