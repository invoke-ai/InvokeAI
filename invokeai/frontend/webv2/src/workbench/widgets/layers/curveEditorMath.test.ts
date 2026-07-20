import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';

import { describe, expect, it, vi } from 'vitest';

import { curvePointFromSvg, curvePointToSvg, finishCurveDragResult, getCurveGridCoordinates } from './curveEditorMath';

const BEFORE: CanvasAdjustmentsContract = {
  brightness: 0,
  contrast: 0,
  curves: {
    b: [
      [0, 0],
      [255, 255],
    ],
    g: [
      [0, 0],
      [255, 255],
    ],
    r: [
      [0, 0],
      [255, 255],
    ],
  },
  saturation: 0,
};

const CURRENT: CanvasAdjustmentsContract = {
  ...BEFORE,
  curves: {
    ...BEFORE.curves!,
    r: [
      [0, 32],
      [255, 255],
    ],
  },
};

describe('curve editor coordinates', () => {
  it('insets endpoints so handles are fully visible', () => {
    expect(curvePointToSvg(0, 0)).toEqual({ cx: 6, cy: 174 });
    expect(curvePointToSvg(255, 255)).toEqual({ cx: 174, cy: 6 });
  });

  it('round-trips and clamps pointer coordinates through the inset plot', () => {
    expect(curvePointFromSvg(90, 90)).toEqual([128, 128]);
    expect(curvePointFromSvg(-20, 220)).toEqual([0, 0]);
    expect(curvePointFromSvg(220, -20)).toEqual([255, 255]);
  });

  it('places the grid at quarter intervals within the inset plot', () => {
    expect(getCurveGridCoordinates()).toEqual([6, 48, 90, 132, 174]);
  });

  it('restores the pre-drag snapshot without committing when cancelled after a live change', () => {
    const onPreview = vi.fn();
    const onCommit = vi.fn();

    finishCurveDragResult({ before: BEFORE, cancelled: true, current: CURRENT, onCommit, onPreview });

    expect(onPreview).toHaveBeenCalledTimes(1);
    expect(onPreview).toHaveBeenCalledWith(BEFORE);
    expect(onCommit).not.toHaveBeenCalled();
  });

  it('commits the current snapshot without restoring on pointer up', () => {
    const onPreview = vi.fn();
    const onCommit = vi.fn();

    finishCurveDragResult({ before: BEFORE, cancelled: false, current: CURRENT, onCommit, onPreview });

    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit).toHaveBeenCalledWith(CURRENT);
    expect(onPreview).not.toHaveBeenCalled();
  });
});
