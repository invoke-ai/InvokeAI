import { describe, expect, it } from 'vitest';

import { getCanvasToolModifierHintIds } from './canvasToolModifierHints';

describe('getCanvasToolModifierHintIds', () => {
  it('returns brush hints in priority order', () => {
    expect(
      getCanvasToolModifierHintIds({
        tool: 'brush',
        lassoMode: 'freehand',
        bboxAspectRatioLocked: false,
        hasActiveTextSession: false,
      })
    ).toEqual(['shiftStraightLine', 'modWheelResizeBrush', 'spacePan', 'altPickColor']);
  });

  it('omits alt color-picker hint for eraser', () => {
    expect(
      getCanvasToolModifierHintIds({
        tool: 'eraser',
        lassoMode: 'freehand',
        bboxAspectRatioLocked: false,
        hasActiveTextSession: false,
      })
    ).toEqual(['shiftStraightLine', 'modWheelResizeEraser', 'spacePan']);
  });

  it('adds polygon snapping for polygon lasso', () => {
    expect(
      getCanvasToolModifierHintIds({
        tool: 'lasso',
        lassoMode: 'polygon',
        bboxAspectRatioLocked: false,
        hasActiveTextSession: false,
      })
    ).toEqual(['modSubtractMask', 'shiftSnap45Degrees', 'spacePan']);
  });

  it('omits polygon snapping for freehand lasso', () => {
    expect(
      getCanvasToolModifierHintIds({
        tool: 'lasso',
        lassoMode: 'freehand',
        bboxAspectRatioLocked: false,
        hasActiveTextSession: false,
      })
    ).toEqual(['modSubtractMask', 'spacePan']);
  });

  it('switches the bbox aspect-ratio hint based on lock state', () => {
    expect(
      getCanvasToolModifierHintIds({
        tool: 'bbox',
        lassoMode: 'freehand',
        bboxAspectRatioLocked: false,
        hasActiveTextSession: false,
      })
    ).toEqual(['shiftLockAspectRatio', 'altScaleFromCenter', 'modFineGrid']);

    expect(
      getCanvasToolModifierHintIds({
        tool: 'bbox',
        lassoMode: 'freehand',
        bboxAspectRatioLocked: true,
        hasActiveTextSession: false,
      })
    ).toEqual(['shiftUnlockAspectRatio', 'altScaleFromCenter', 'modFineGrid']);
  });

  it('only shows text-session hints when a text session is active', () => {
    expect(
      getCanvasToolModifierHintIds({
        tool: 'text',
        lassoMode: 'freehand',
        bboxAspectRatioLocked: false,
        hasActiveTextSession: true,
      })
    ).toEqual(['enterCommitText', 'shiftEnterNewLine', 'escCancelText', 'modDragText', 'shiftSnapRotation']);

    expect(
      getCanvasToolModifierHintIds({
        tool: 'text',
        lassoMode: 'freehand',
        bboxAspectRatioLocked: false,
        hasActiveTextSession: false,
      })
    ).toEqual(['spacePan', 'altPickColor']);
  });
});
