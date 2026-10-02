import type {
  CanvasInpaintMaskLayerContract,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import { isConfigApplied } from './postconditions';

const regional = (
  overrides: Partial<CanvasRegionalGuidanceLayerContract> = {}
): CanvasRegionalGuidanceLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id: 'rg',
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#ff0000', style: 'solid' } },
  name: 'Region',
  negativePrompt: null,
  opacity: 0.5,
  positivePrompt: null,
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
  ...overrides,
});

const inpaint = (overrides: Partial<CanvasInpaintMaskLayerContract> = {}): CanvasInpaintMaskLayerContract => ({
  blendMode: 'normal',
  id: 'mask',
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#ff0000', style: 'solid' } },
  name: 'Mask',
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
  ...overrides,
});

describe('isConfigApplied', () => {
  it('accepts a cleared prompt, which the layer stores as null', () => {
    expect(isConfigApplied(regional(), { layerType: 'regional_guidance', negativePrompt: null })).toBe(true);
    expect(isConfigApplied(regional(), { layerType: 'regional_guidance', positivePrompt: null })).toBe(true);
  });

  it('rejects a prompt that still holds text after a clear', () => {
    expect(
      isConfigApplied(regional({ negativePrompt: 'blurry' }), { layerType: 'regional_guidance', negativePrompt: null })
    ).toBe(false);
  });

  it('matches a set prompt by value', () => {
    expect(
      isConfigApplied(regional({ negativePrompt: 'blurry' }), {
        layerType: 'regional_guidance',
        negativePrompt: 'blurry',
      })
    ).toBe(true);
    expect(
      isConfigApplied(regional({ negativePrompt: 'blurry' }), {
        layerType: 'regional_guidance',
        negativePrompt: 'sharp',
      })
    ).toBe(false);
  });

  it('accepts a removed mask modifier, which the reducer deletes', () => {
    expect(isConfigApplied(inpaint(), { layerType: 'inpaint_mask', noise: null })).toBe(true);
    expect(
      isConfigApplied(inpaint({ noise: { isEnabled: true, level: 0.5 } }), { layerType: 'inpaint_mask', noise: null })
    ).toBe(false);
  });
});
