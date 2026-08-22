import { describe, expect, it } from 'vitest';

import { getInvokeIconMode } from './invokeButtonModel';

describe('getInvokeIconMode', () => {
  it('shows play when the queue is idle', () => {
    expect(getInvokeIconMode({ hasOpenWork: false, isHovered: false, progress: null })).toEqual({ mode: 'play' });
  });

  it('shows progress while running and not hovered', () => {
    expect(getInvokeIconMode({ hasOpenWork: true, isHovered: false, progress: 0.4 })).toEqual({
      mode: 'progress',
      value: 0.4,
    });
  });

  it('shows indeterminate progress while running before the first progress event', () => {
    expect(getInvokeIconMode({ hasOpenWork: true, isHovered: false, progress: null })).toEqual({
      mode: 'progress',
      value: null,
    });
  });

  it('reverts to play on hover so queueing more work reads as available', () => {
    expect(getInvokeIconMode({ hasOpenWork: true, isHovered: true, progress: 0.4 })).toEqual({ mode: 'play' });
  });
});
