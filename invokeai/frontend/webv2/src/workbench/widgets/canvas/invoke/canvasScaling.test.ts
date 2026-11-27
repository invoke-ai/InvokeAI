import { describe, expect, it } from 'vitest';

import { readCanvasScaling, readLockedAspectRatio } from './canvasScaling';

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

describe('readLockedAspectRatio', () => {
  it('holds a preset ratio, which implies the lock', () => {
    expect(readLockedAspectRatio({ aspectRatioId: '16:9', aspectRatioIsLocked: false, aspectRatioValue: 16 / 9 })).toBe(
      16 / 9
    );
  });

  it('holds a free ratio only while it is locked', () => {
    expect(readLockedAspectRatio({ aspectRatioId: 'Free', aspectRatioIsLocked: true, aspectRatioValue: 1.3 })).toBe(
      1.3
    );
    expect(
      readLockedAspectRatio({ aspectRatioId: 'Free', aspectRatioIsLocked: false, aspectRatioValue: 1.3 })
    ).toBeNull();
    expect(readLockedAspectRatio({ aspectRatioValue: 1.3 })).toBeNull();
  });

  it('holds nothing without a usable ratio', () => {
    expect(readLockedAspectRatio({ aspectRatioId: '1:1', aspectRatioValue: 0 })).toBeNull();
    expect(readLockedAspectRatio({ aspectRatioId: '1:1', aspectRatioValue: 'wide' })).toBeNull();
  });
});
