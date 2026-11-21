/**
 * Tells a notched mouse wheel from smooth scrolling per wheel event. Notch signatures: line or page `deltaMode`
 * (Firefox); a vertical pixel delta that is a multiple of 4.000244140625 (macOS non-precise wheel in Chromium and
 * Safari, as in MapLibre); a legacy `wheelDeltaY` multiple of 120 with a delta of 25px or more (Chromium, Safari;
 * skipped for a pinch, whose every step Chromium stamps 120 too); a first delta of 50px or more (Chromium on Wayland
 * reports a notch as 150px, wheelDelta 180). Trackpads start with small deltas and often move on both axes at once,
 * which no wheel does. Events at most 200ms apart form a stream that keeps its kind unless a line-mode, macOS-notch
 * or two-axis event says otherwise, so a flick's one large delta does not flip it.
 */

import type { Size } from '@workbench/canvas-engine/viewport';

export type WheelDeviceKind = 'notched' | 'smooth';

/** The fields read, with Chromium/Safari's legacy `wheelDeltaY`, which lib.dom omits. */
export interface WheelEventShape {
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
}

const DOM_DELTA_LINE = 1;
const DOM_DELTA_PAGE = 2;
const WHEEL_LINE_PX = 16;
const MAC_NOTCH_PX = 4.000244140625;
const LEGACY_NOTCH = 120;
const LEGACY_NOTCH_MIN_PX = 25;
const FIRST_NOTCH_MIN_PX = 50;
const WHEEL_STREAM_GAP_MS = 200;

/** `pinch`: ctrl+wheel with the Control key up, whose `wheelDeltaY` Chromium stamps as a whole tick. */
export const createWheelDeviceClassifier = (): ((
  event: WheelEventShape,
  viewportSize: Size,
  pinch: boolean
) => WheelSample) => {
  let streamKind: WheelDeviceKind = 'smooth';
  let lastTimeStamp = Number.NEGATIVE_INFINITY;

  return (event, viewportSize, pinch) => {
    // Firefox reports a line-mode wheel in pixels once a delta is read before `deltaMode`.
    const mode = event.deltaMode;
    const rawX = event.deltaX;
    const rawY = event.deltaY;
    const scaleX = mode === DOM_DELTA_PAGE ? viewportSize.width : mode === DOM_DELTA_LINE ? WHEEL_LINE_PX : 1;
    const scaleY = mode === DOM_DELTA_PAGE ? viewportSize.height : mode === DOM_DELTA_LINE ? WHEEL_LINE_PX : 1;
    const twoAxis = rawX !== 0 && rawY !== 0;
    const sameStream = event.timeStamp - lastTimeStamp <= WHEEL_STREAM_GAP_MS;

    if (mode === DOM_DELTA_LINE || mode === DOM_DELTA_PAGE) {
      streamKind = 'notched';
    } else if (twoAxis) {
      streamKind = 'smooth';
    } else if (rawX === 0 && rawY !== 0 && rawY % MAC_NOTCH_PX === 0) {
      streamKind = 'notched';
    } else if (!sameStream) {
      const wheelDeltaY = event.wheelDeltaY ?? 0;
      const size = Math.abs(rawY);
      const legacyNotch =
        !pinch && wheelDeltaY !== 0 && wheelDeltaY % LEGACY_NOTCH === 0 && size >= LEGACY_NOTCH_MIN_PX;
      const notched = rawX === 0 && (legacyNotch || size >= FIRST_NOTCH_MIN_PX);
      streamKind = notched ? 'notched' : 'smooth';
    }
    lastTimeStamp = event.timeStamp;

    return { deltaX: rawX * scaleX, deltaY: rawY * scaleY, kind: streamKind, twoAxis };
  };
};
