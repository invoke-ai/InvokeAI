import type { WheelEventShape } from '@workbench/canvas-engine/input/wheelDevice';

import { createWheelDeviceClassifier } from '@workbench/canvas-engine/input/wheelDevice';
import { describe, expect, it } from 'vitest';

const VIEWPORT = { height: 600, width: 800 };

const wheel = (deltaY: number, init: Partial<WheelEventShape> = {}): WheelEventShape => ({
  deltaMode: 0,
  deltaX: 0,
  deltaY,
  timeStamp: 0,
  ...init,
});

describe('wheel device classifier', () => {
  it.each<[string, WheelEventShape, 'notched' | 'smooth']>([
    ['a Chromium notch', wheel(100, { wheelDeltaY: -120 }), 'notched'],
    ['a Firefox line-mode notch', wheel(3, { deltaMode: 1 }), 'notched'],
    ['a macOS non-precise wheel notch', wheel(4.000244140625 * 3, { wheelDeltaY: -36 }), 'notched'],
    ['a Chromium notch on Wayland', wheel(150, { wheelDeltaY: -180 }), 'notched'],
    ['a small trackpad delta', wheel(3, { wheelDeltaY: -9 }), 'smooth'],
    ['a Chromium pinch step, stamped as one tick', wheel(-2.4, { wheelDeltaY: 120 }), 'smooth'],
    ['a two-axis trackpad delta', wheel(6, { deltaX: 1.5, wheelDeltaY: -18 }), 'smooth'],
  ])('reads %s', (_name, event, kind) => {
    expect(createWheelDeviceClassifier()(event, VIEWPORT, false).kind).toBe(kind);
  });

  it('reads a large first pinch step as smooth, but still a notch-sized one as a notch', () => {
    expect(createWheelDeviceClassifier()(wheel(-30, { wheelDeltaY: 120 }), VIEWPORT, true).kind).toBe('smooth');
    expect(createWheelDeviceClassifier()(wheel(-100, { wheelDeltaY: 120 }), VIEWPORT, true).kind).toBe('notched');
  });

  it('reads deltaMode before the deltas, as Firefox reports a line-mode wheel in pixels otherwise', () => {
    let modeRead = false;
    const firefoxWheel: WheelEventShape = {
      get deltaMode() {
        modeRead = true;
        return 1;
      },
      deltaX: 0,
      get deltaY() {
        return modeRead ? 3 : 57;
      },
      timeStamp: 0,
    };
    expect(createWheelDeviceClassifier()(firefoxWheel, VIEWPORT, false)).toMatchObject({ deltaY: 48, kind: 'notched' });
  });

  it('normalizes lines to 16px and pages to the viewport size', () => {
    const classify = createWheelDeviceClassifier();
    expect(classify(wheel(3, { deltaMode: 1, deltaX: -1 }), VIEWPORT, false)).toMatchObject({
      deltaX: -16,
      deltaY: 48,
    });
    expect(classify(wheel(1, { deltaMode: 2, deltaX: 0.5 }), VIEWPORT, false)).toMatchObject({
      deltaX: 400,
      deltaY: 600,
    });
  });

  it('keeps a trackpad stream smooth through a notch-looking delta, and re-judges after a 200ms pause', () => {
    const classify = createWheelDeviceClassifier();
    expect(classify(wheel(3, { timeStamp: 0 }), VIEWPORT, false).kind).toBe('smooth');
    expect(classify(wheel(60, { timeStamp: 16, wheelDeltaY: -120 }), VIEWPORT, false).kind).toBe('smooth');
    expect(classify(wheel(60, { timeStamp: 216, wheelDeltaY: -120 }), VIEWPORT, false).kind).toBe('smooth');
    expect(classify(wheel(60, { timeStamp: 420, wheelDeltaY: -120 }), VIEWPORT, false).kind).toBe('notched');
  });

  it('turns a notched stream smooth on a two-axis event', () => {
    const classify = createWheelDeviceClassifier();
    expect(classify(wheel(100, { wheelDeltaY: -120 }), VIEWPORT, false).kind).toBe('notched');
    expect(classify(wheel(4, { deltaX: 2, timeStamp: 16 }), VIEWPORT, false).kind).toBe('smooth');
    expect(classify(wheel(4, { timeStamp: 32 }), VIEWPORT, false).kind).toBe('smooth');
  });
});
