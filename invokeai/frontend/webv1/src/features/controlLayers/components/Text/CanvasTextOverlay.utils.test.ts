import { describe, expect, it } from 'vitest';

import { getCanvasTextEditorEffectiveSize, getInitialCanvasTextEditorSize } from './CanvasTextOverlay.utils';

describe('CanvasTextOverlay utils', () => {
  it('computes the initial editor size from measured content metrics and padding', () => {
    const size = getInitialCanvasTextEditorSize({
      contentMetrics: { contentWidth: 120, contentHeight: 24 },
      textContainerData: {
        padding: 16,
        extraLeftPadding: 3,
        extraRightPadding: 7,
      },
      minSize: 32,
    });

    expect(size).toEqual({ width: 162, height: 56 });
  });

  it('uses the content metrics size before ResizeObserver has reported a measured size', () => {
    const size = getCanvasTextEditorEffectiveSize({
      measuredSize: null,
      contentMetrics: { contentWidth: 180, contentHeight: 40 },
      textContainerData: {
        padding: 12,
        extraLeftPadding: 4,
        extraRightPadding: 6,
      },
      minSize: 20,
    });

    expect(size).toEqual({ width: 214, height: 64 });
  });

  it('uses the observed measured size after ResizeObserver reports one', () => {
    const size = getCanvasTextEditorEffectiveSize({
      measuredSize: { width: 240, height: 96 },
      contentMetrics: { contentWidth: 180, contentHeight: 40 },
      textContainerData: {
        padding: 12,
        extraLeftPadding: 4,
        extraRightPadding: 6,
      },
      minSize: 20,
    });

    expect(size).toEqual({ width: 240, height: 96 });
  });
});
