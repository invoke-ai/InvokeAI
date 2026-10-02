import { describe, expect, it } from 'vitest';

import { readCanvasScaling } from './canvasScaling';

describe('readCanvasScaling', () => {
  it('defaults to no scaling with no manual size', () => {
    expect(readCanvasScaling(undefined)).toEqual({ height: null, method: 'none', width: null });
    expect(readCanvasScaling({ scaleMethod: 'bigger', scaledWidth: -5, scaledHeight: 'tall' })).toEqual({
      height: null,
      method: 'none',
      width: null,
    });
  });

  it('reads a persisted manual size as whole pixels', () => {
    expect(readCanvasScaling({ scaleMethod: 'manual', scaledWidth: 768.4, scaledHeight: 512 })).toEqual({
      height: 512,
      method: 'manual',
      width: 768,
    });
  });
});
