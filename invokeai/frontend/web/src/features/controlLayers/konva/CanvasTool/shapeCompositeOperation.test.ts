import type { CanvasEntityState } from 'features/controlLayers/store/types';
import { describe, expect, it } from 'vitest';

import { getShapeCompositeOperation } from './shapeCompositeOperation';

describe('shape composite operation', () => {
  it('uses destination-out for subtractive drawing even when transparency is locked', () => {
    const entity = {
      type: 'raster_layer',
      isTransparencyLocked: true,
    } as CanvasEntityState;

    expect(getShapeCompositeOperation(entity, true)).toBe('destination-out');
  });

  it('uses source-atop for normal drawing on locked raster layers', () => {
    const entity = {
      type: 'raster_layer',
      isTransparencyLocked: true,
    } as CanvasEntityState;

    expect(getShapeCompositeOperation(entity, false)).toBe('source-atop');
  });

  it('uses source-over for normal drawing when transparency is not locked', () => {
    const entity = {
      type: 'raster_layer',
      isTransparencyLocked: false,
    } as CanvasEntityState;

    expect(getShapeCompositeOperation(entity, false)).toBe('source-over');
    expect(getShapeCompositeOperation(null, false)).toBe('source-over');
  });
});
