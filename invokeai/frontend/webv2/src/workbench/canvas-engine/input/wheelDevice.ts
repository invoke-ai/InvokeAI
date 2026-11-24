/**
 * Tells a notched mouse wheel from smooth scrolling per wheel event. Notch signatures: line or page `deltaMode`
 * (Firefox); a vertical pixel delta that is a multiple of 4.000244140625 (macOS non-precise wheel in Chromium and
 * Safari, as in MapLibre); a legacy `wheelDeltaY` multiple of 120 with a delta of 25px or more (Chromium, Safari;
 * skipped for a pinch); a delta of 50px or more (Chromium on Wayland reports a notch as 150px, wheelDelta 180).
 * Trackpads start with small deltas and often move on both axes at once, which no wheel does. Events at most 200ms
 * apart form a stream, judged by its first vertical one: a smooth stream stays smooth through a flick's large delta
 * unless a line-mode or macOS-notch event says otherwise; a notched one turns smooth on a two-axis event or on any
 * vertical event that is not notch-shaped. A pinch is a smooth ctrl+wheel with Control known up or, while its state
 * is unknown, one carrying Chromium's pinch stamp: a whole-tick `wheelDeltaY` on a delta under 25px.
 */

import type { ControlKeyState } from '@workbench/canvas-engine/input/pointerPipeline';
import type { Size } from '@workbench/canvas-engine/viewport';

export type WheelDeviceKind = 'notched' | 'smooth';

/** The fields read, with Chromium/Safari's legacy `wheelDeltaY`, which lib.dom omits. */
export interface WheelEventShape {
  readonly ctrlKey: boolean;
  readonly deltaMode: number;
  readonly deltaX: number;
  readonly deltaY: number;
  readonly timeStamp: number;
  readonly wheelDeltaY?: number;
}

export interface WheelSample {
  kind: WheelDeviceKind;
  /** Deltas in CSS pixels. */
  deltaX: number;
  deltaY: number;
  /** Moved on both axes at once: only a trackpad does that. */
  twoAxis: boolean;
  /** A trackpad pinch: a smooth ctrl+wheel that a held Control key does not explain. */
  pinch: boolean;
}

const DOM_DELTA_PIXEL = 0;
const DOM_DELTA_LINE = 1;
const DOM_DELTA_PAGE = 2;
const WHEEL_LINE_PX = 16;
const MAC_NOTCH_PX = 4.000244140625;
const LEGACY_NOTCH = 120;
const LEGACY_NOTCH_MIN_PX = 25;
const NOTCH_MIN_PX = 50;
const WHEEL_STREAM_GAP_MS = 200;

/** `controlKey`: the physical Control key, which tells a pinch's ctrl+wheel from a held Control. */
export const createWheelDeviceClassifier = (): ((
  event: WheelEventShape,
  viewportSize: Size,
  controlKey: ControlKeyState
) => WheelSample) => {
  let streamKind: WheelDeviceKind = 'smooth';
  let lastTimeStamp = Number.NEGATIVE_INFINITY;

  return (event, viewportSize, controlKey) => {
    // Firefox reports a line-mode wheel in pixels once a delta is read before `deltaMode`.
    const mode = event.deltaMode;
    const rawX = event.deltaX;
    const rawY = event.deltaY;
    const wheelDeltaY = event.wheelDeltaY ?? 0;
    const scaleX = mode === DOM_DELTA_PAGE ? viewportSize.width : mode === DOM_DELTA_LINE ? WHEEL_LINE_PX : 1;
    const scaleY = mode === DOM_DELTA_PAGE ? viewportSize.height : mode === DOM_DELTA_LINE ? WHEEL_LINE_PX : 1;
    const size = Math.abs(rawY);
    const twoAxis = rawX !== 0 && rawY !== 0;
    const vertical = rawX === 0 && rawY !== 0;
    const macNotch = vertical && rawY % MAC_NOTCH_PX === 0;
    const wholeTick = wheelDeltaY !== 0 && wheelDeltaY % LEGACY_NOTCH === 0;
    const pinchStamp = mode === DOM_DELTA_PIXEL && vertical && !macNotch && wholeTick && size < LEGACY_NOTCH_MIN_PX;
    const pinchCandidate = event.ctrlKey && (controlKey === 'up' || (controlKey === 'unknown' && pinchStamp));
    const legacyNotch = !pinchCandidate && wholeTick && size >= LEGACY_NOTCH_MIN_PX;
    const notchShaped = legacyNotch || size >= NOTCH_MIN_PX;
    const sameStream = event.timeStamp - lastTimeStamp <= WHEEL_STREAM_GAP_MS;

    if (mode === DOM_DELTA_LINE || mode === DOM_DELTA_PAGE) {
      streamKind = 'notched';
    } else if (twoAxis) {
      streamKind = 'smooth';
    } else if (macNotch) {
      streamKind = 'notched';
    } else if (vertical && !(sameStream && streamKind === 'smooth')) {
      streamKind = notchShaped ? 'notched' : 'smooth';
    }
    // A sideways-only or empty event, as a tilt click around a spin, extends a stream but never opens one.
    if (rawY !== 0 || sameStream) {
      lastTimeStamp = event.timeStamp;
    }

    return {
      deltaX: rawX * scaleX,
      deltaY: rawY * scaleY,
      kind: streamKind,
      pinch: pinchCandidate && streamKind === 'smooth',
      twoAxis,
    };
  };
};
