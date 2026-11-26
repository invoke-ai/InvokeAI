import type { WheelEventShape } from '@workbench/canvas-engine/input/wheelDevice';

import { createWheelDeviceClassifier } from '@workbench/canvas-engine/input/wheelDevice';
import { describe, expect, it } from 'vitest';

const VIEWPORT = { height: 600, width: 800 };

const wheel = (deltaY: number, init: Partial<WheelEventShape> = {}): WheelEventShape => ({
  ctrlKey: false,
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
    expect(createWheelDeviceClassifier()(event, VIEWPORT, 'down').kind).toBe(kind);
  });

  it('reads a large first pinch step as smooth, but still a notch-sized one as a notch', () => {
    const pinch = { ctrlKey: true, wheelDeltaY: 120 };
    expect(createWheelDeviceClassifier()(wheel(-30, pinch), VIEWPORT, 'up')).toMatchObject({
      kind: 'smooth',
      pinch: true,
    });
    expect(createWheelDeviceClassifier()(wheel(-100, pinch), VIEWPORT, 'up')).toMatchObject({
      kind: 'notched',
      pinch: false,
    });
  });

  it('reads no pinch, with Control unknown, without ctrl or without Chromium’s whole-tick stamp', () => {
    const pinch = (event: WheelEventShape) => createWheelDeviceClassifier()(event, VIEWPORT, 'unknown').pinch;
    expect(pinch(wheel(-2.4, { wheelDeltaY: 120 }))).toBe(false);
    expect(pinch(wheel(-4, { ctrlKey: true, wheelDeltaY: 12 }))).toBe(false);
  });

  it('reads deltaMode before the deltas, as Firefox reports a line-mode wheel in pixels otherwise', () => {
    let modeRead = false;
    const firefoxWheel: WheelEventShape = {
      get deltaMode() {
        modeRead = true;
        return 1;
      },
      ctrlKey: false,
      deltaX: 0,
      get deltaY() {
        return modeRead ? 3 : 57;
      },
      timeStamp: 0,
    };
    expect(createWheelDeviceClassifier()(firefoxWheel, VIEWPORT, 'down')).toMatchObject({
      deltaY: 48,
      kind: 'notched',
    });
  });

  it('normalizes lines to 16px and pages to the viewport size', () => {
    const classify = createWheelDeviceClassifier();
    expect(classify(wheel(3, { deltaMode: 1, deltaX: -1 }), VIEWPORT, 'down')).toMatchObject({
      deltaX: -16,
      deltaY: 48,
    });
    expect(classify(wheel(1, { deltaMode: 2, deltaX: 0.5 }), VIEWPORT, 'down')).toMatchObject({
      deltaX: 400,
      deltaY: 600,
    });
  });

  it('keeps a trackpad stream smooth through a notch-looking delta, and re-judges after a 200ms pause', () => {
    const classify = createWheelDeviceClassifier();
    expect(classify(wheel(3, { timeStamp: 0 }), VIEWPORT, 'down').kind).toBe('smooth');
    expect(classify(wheel(60, { timeStamp: 16, wheelDeltaY: -120 }), VIEWPORT, 'down').kind).toBe('smooth');
    expect(classify(wheel(60, { timeStamp: 216, wheelDeltaY: -120 }), VIEWPORT, 'down').kind).toBe('smooth');
    expect(classify(wheel(60, { timeStamp: 420, wheelDeltaY: -120 }), VIEWPORT, 'down').kind).toBe('notched');
  });

  it.each<[string, [number, number][], ('notched' | 'smooth')[]]>([
    [
      'a flick that starts at 50px or more',
      [
        [60, -180],
        [50, -150],
        [30, -90],
        [12, -36],
        [5, -15],
      ],
      ['notched', 'notched', 'smooth', 'smooth', 'smooth'],
    ],
    [
      'a flick that starts with a whole tick',
      [
        [40, -120],
        [20, -60],
        [8, -24],
      ],
      ['notched', 'smooth', 'smooth'],
    ],
  ])(
    'turns a notched stream smooth on the first vertical event that is not notch-shaped: %s',
    (_name, events, kinds) => {
      const classify = createWheelDeviceClassifier();
      const read = events.map(
        ([deltaY, wheelDeltaY], index) =>
          classify(wheel(deltaY, { timeStamp: index * 16, wheelDeltaY }), VIEWPORT, 'down').kind
      );
      expect(read).toEqual(kinds);
    }
  );

  it('judges a stream by its vertical events only, so a tilt click before or inside a spin leaves it notched', () => {
    const classify = createWheelDeviceClassifier();
    const tilt = (timeStamp: number) => classify(wheel(0, { deltaX: 30, timeStamp }), VIEWPORT, 'down');
    const notch = (timeStamp: number) => classify(wheel(150, { timeStamp, wheelDeltaY: -180 }), VIEWPORT, 'down').kind;
    tilt(0);
    expect([notch(16), notch(66)]).toEqual(['notched', 'notched']);
    tilt(100);
    expect(notch(150)).toBe('notched');
  });

  it('keeps a trackpad stream open through a sideways-only swipe', () => {
    const classify = createWheelDeviceClassifier();
    classify(wheel(3, { timeStamp: 0 }), VIEWPORT, 'down');
    for (const timeStamp of [150, 300]) {
      classify(wheel(0, { deltaX: 20, timeStamp }), VIEWPORT, 'down');
    }
    expect(classify(wheel(60, { timeStamp: 316, wheelDeltaY: -180 }), VIEWPORT, 'down').kind).toBe('smooth');
  });

  it('turns a notched stream smooth on a two-axis event', () => {
    const classify = createWheelDeviceClassifier();
    expect(classify(wheel(100, { wheelDeltaY: -120 }), VIEWPORT, 'down').kind).toBe('notched');
    expect(classify(wheel(4, { deltaX: 2, timeStamp: 16 }), VIEWPORT, 'down').kind).toBe('smooth');
    expect(classify(wheel(4, { timeStamp: 32 }), VIEWPORT, 'down').kind).toBe('smooth');
  });
});
