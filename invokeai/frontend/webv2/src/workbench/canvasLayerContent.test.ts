import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/api';

import { describe, expect, it } from 'vitest';

import {
  hasControlLayerContent,
  hasRegionalGuidanceMaskContent,
  isCompositableControlLayer,
  isCompositableRegionalGuidanceLayer,
} from './canvasLayerContent';

const controlLayer = (
  id: string,
  overrides: { isEnabled?: boolean; source?: CanvasControlLayerContract['source'] } = {}
): CanvasControlLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: null, kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  id,
  isEnabled: overrides.isEnabled ?? true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: overrides.source ?? { image: { height: 64, imageName: `${id}.png`, width: 64 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

const regionalLayer = (
  id: string,
  overrides: { isEnabled?: boolean; bitmap?: CanvasRegionalGuidanceLayerContract['mask']['bitmap'] } = {}
): CanvasRegionalGuidanceLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id,
  isEnabled: overrides.isEnabled ?? true,
  isLocked: false,
  mask: {
    bitmap: overrides.bitmap === undefined ? ({ height: 8, width: 8 } as never) : overrides.bitmap,
    fill: { color: '#ffffff', style: 'solid' },
    offset: { x: 0, y: 0 },
  },
  name: id,
  negativePrompt: null,
  opacity: 1,
  positivePrompt: null,
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
});

describe('hasControlLayerContent', () => {
  it('accepts an image source and committed paint pixels, rejects blank paint', () => {
    expect(hasControlLayerContent(controlLayer('image'))).toBe(true);
    expect(
      hasControlLayerContent(
        controlLayer('painted', { source: { bitmap: { height: 8, width: 8 } as never, type: 'paint' } })
      )
    ).toBe(true);
    expect(hasControlLayerContent(controlLayer('blank', { source: { bitmap: null, type: 'paint' } }))).toBe(false);
  });
});

describe('isCompositableControlLayer', () => {
  it('requires the control type, enablement, and content together', () => {
    expect(isCompositableControlLayer(controlLayer('ready'))).toBe(true);
    expect(isCompositableControlLayer(controlLayer('disabled', { isEnabled: false }))).toBe(false);
    expect(isCompositableControlLayer(controlLayer('blank', { source: { bitmap: null, type: 'paint' } }))).toBe(false);
    expect(isCompositableControlLayer(regionalLayer('not-control') as CanvasLayerContract)).toBe(false);
  });
});

describe('hasRegionalGuidanceMaskContent', () => {
  it('requires a persisted mask bitmap', () => {
    expect(hasRegionalGuidanceMaskContent(regionalLayer('masked'))).toBe(true);
    expect(hasRegionalGuidanceMaskContent(regionalLayer('empty', { bitmap: null }))).toBe(false);
  });
});

describe('isCompositableRegionalGuidanceLayer', () => {
  it('requires the regional type, enablement, and a mask together', () => {
    expect(isCompositableRegionalGuidanceLayer(regionalLayer('ready'))).toBe(true);
    expect(isCompositableRegionalGuidanceLayer(regionalLayer('disabled', { isEnabled: false }))).toBe(false);
    expect(isCompositableRegionalGuidanceLayer(regionalLayer('empty', { bitmap: null }))).toBe(false);
    expect(isCompositableRegionalGuidanceLayer(controlLayer('not-regional') as CanvasLayerContract)).toBe(false);
  });
});
