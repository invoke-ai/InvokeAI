import { describe, expect, it } from 'vitest';

import { getInvokeIconMode } from './invokeButtonModel';

describe('getInvokeIconMode', () => {
  it.each([
    ['shows play when the queue is idle', false, false, false, null, { mode: 'play' }],
    ['shows progress while running and not hovered', true, false, false, 0.4, { mode: 'progress', value: 0.4 }],
    [
      'shows indeterminate progress while running before the first progress event',
      true,
      false,
      false,
      null,
      { mode: 'progress', value: null },
    ],
    ['reverts to play on hover so queueing more work reads as available', true, false, true, 0.4, { mode: 'play' }],
    [
      'shows the front-of-queue glyph while Shift is held over an idle button',
      false,
      true,
      true,
      null,
      { mode: 'front' },
    ],
    [
      'shows the front-of-queue glyph while Shift is held over a running button',
      true,
      true,
      true,
      0.4,
      { mode: 'front' },
    ],
    [
      'keeps play for an idle button that is not hovered while Shift is held',
      false,
      true,
      false,
      null,
      { mode: 'play' },
    ],
    [
      'keeps progress for a running button that is not hovered while Shift is held',
      true,
      true,
      false,
      0.4,
      { mode: 'progress', value: 0.4 },
    ],
  ])('%s', (_, hasOpenWork, isFrontHeld, isHovered, progress, expected) => {
    expect(getInvokeIconMode({ hasOpenWork, isFrontHeld, isHovered, progress })).toEqual(expected);
  });
});
