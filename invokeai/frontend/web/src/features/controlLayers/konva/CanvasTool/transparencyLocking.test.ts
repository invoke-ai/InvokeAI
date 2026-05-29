import type { CanvasEntityState } from 'features/controlLayers/store/types';
import { describe, expect, it } from 'vitest';

import { getTransparencyLockedCompositeOperation } from './transparencyLocking';

describe('transparency locking', () => {
  it('uses source-atop for locked raster layers', () => {
    const entity = {
      type: 'raster_layer',
      isTransparencyLocked: true,
    } as CanvasEntityState;

    expect(getTransparencyLockedCompositeOperation(entity)).toBe('source-atop');
  });

  it('does not change compositing for unlocked or non-raster entities', () => {
    const unlockedRasterLayer = {
      type: 'raster_layer',
      isTransparencyLocked: false,
    } as CanvasEntityState;
    const controlLayer = {
      type: 'control_layer',
      isTransparencyLocked: true,
    } as CanvasEntityState;

    expect(getTransparencyLockedCompositeOperation(unlockedRasterLayer)).toBeUndefined();
    expect(getTransparencyLockedCompositeOperation(controlLayer)).toBeUndefined();
    expect(getTransparencyLockedCompositeOperation(null)).toBeUndefined();
  });
});
