import { describe, expect, it } from 'vitest';

import { getInvokeIconMode } from './invokeButtonModel';

describe('getInvokeIconMode', () => {
  it('shows play when the queue is idle', () => {
    expect(getInvokeIconMode({ hasOpenWork: false, isFrontHeld: false, isHovered: false, progress: null })).toEqual({
      mode: 'play',
    });
  });

  it('shows progress while running and not hovered', () => {
    expect(getInvokeIconMode({ hasOpenWork: true, isFrontHeld: false, isHovered: false, progress: 0.4 })).toEqual({
      mode: 'progress',
      value: 0.4,
    });
  });

  it('shows indeterminate progress while running before the first progress event', () => {
    expect(getInvokeIconMode({ hasOpenWork: true, isFrontHeld: false, isHovered: false, progress: null })).toEqual({
      mode: 'progress',
      value: null,
    });
  });

  it('reverts to play on hover so queueing more work reads as available', () => {
    expect(getInvokeIconMode({ hasOpenWork: true, isFrontHeld: false, isHovered: true, progress: 0.4 })).toEqual({
      mode: 'play',
    });
  });

  it('shows the front-of-queue glyph only while Shift is held over a hovered or focused button', () => {
    expect(getInvokeIconMode({ hasOpenWork: false, isFrontHeld: true, isHovered: true, progress: null })).toEqual({
      mode: 'front',
    });
    expect(getInvokeIconMode({ hasOpenWork: true, isFrontHeld: true, isHovered: true, progress: 0.4 })).toEqual({
      mode: 'front',
    });
    // Not hovered: an idle button keeps play, a running one keeps its progress.
    expect(getInvokeIconMode({ hasOpenWork: false, isFrontHeld: true, isHovered: false, progress: null })).toEqual({
      mode: 'play',
    });
    expect(getInvokeIconMode({ hasOpenWork: true, isFrontHeld: true, isHovered: false, progress: 0.4 })).toEqual({
      mode: 'progress',
      value: 0.4,
    });
  });
});
