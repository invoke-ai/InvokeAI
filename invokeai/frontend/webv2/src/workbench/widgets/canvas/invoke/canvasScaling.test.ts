import { describe, expect, it } from 'vitest';

import { readCanvasScaling } from './canvasScaling';

describe('readCanvasScaling', () => {
  it('defaults to auto scaling with no manual size', () => {
    expect(readCanvasScaling(undefined)).toEqual({ height: null, method: 'auto', width: null });
    expect(readCanvasScaling({ scaleMethod: 'bigger', scaledWidth: -5, scaledHeight: 'tall' })).toEqual({
      height: null,
      method: 'auto',
      width: null,
    });
  });

  it('keeps an explicit choice of no scaling', () => {
    expect(readCanvasScaling({ scaleMethod: 'none' }).method).toBe('none');
  });

  it('reads a persisted manual size as whole pixels', () => {
    expect(readCanvasScaling({ scaleMethod: 'manual', scaledWidth: 768.4, scaledHeight: 512 })).toEqual({
      height: 512,
      method: 'manual',
      width: 768,
    });
  });
});
