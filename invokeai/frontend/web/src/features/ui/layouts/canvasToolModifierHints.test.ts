import { describe, expect, it } from 'vitest';

import { getCanvasToolModifierHintIds } from './canvasToolModifierHints';

const buildArgs = (overrides: Partial<Parameters<typeof getCanvasToolModifierHintIds>[0]> = {}) => ({
  tool: 'brush' as const,
  lassoMode: 'freehand' as const,
  shapeType: 'rect' as const,
  bboxAspectRatioLocked: false,
  hasActiveTextSession: false,
  isPrimaryPointerDown: false,
  ...overrides,
});

describe('getCanvasToolModifierHintIds', () => {
  it('returns brush hints in priority order', () => {
    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'brush' }))).toEqual([
      'shiftStraightLine',
      'modWheelResizeBrush',
      'spacePan',
      'altPickColor',
    ]);
  });

  it('omits alt color-picker hint for eraser', () => {
    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'eraser' }))).toEqual([
      'shiftStraightLine',
      'modWheelResizeEraser',
      'spacePan',
    ]);
  });

  it('adds polygon snapping for polygon lasso', () => {
    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'lasso', lassoMode: 'polygon' }))).toEqual([
      'modErase',
      'shiftSnap45Degrees',
      'spacePan',
    ]);
  });

  it('omits polygon snapping for freehand lasso', () => {
    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'lasso' }))).toEqual(['modErase', 'spacePan']);
  });

  it('switches the bbox aspect-ratio hint based on lock state', () => {
    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'bbox' }))).toEqual([
      'shiftLockAspectRatio',
      'altScaleFromCenter',
      'modFineGrid',
    ]);

    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'bbox', bboxAspectRatioLocked: true }))).toEqual([
      'shiftUnlockAspectRatio',
      'altScaleFromCenter',
      'modFineGrid',
    ]);
  });

  it('only shows text-session hints when a text session is active', () => {
    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'text', hasActiveTextSession: true }))).toEqual([
      'enterCommitText',
      'shiftEnterNewLine',
      'escCancelText',
      'modDragText',
      'shiftSnapRotation',
    ]);

    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'text' }))).toEqual(['spacePan', 'altPickColor']);
  });

  it('shows idle rect and oval shapes hints', () => {
    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'rect', shapeType: 'rect' }))).toEqual([
      'modErase',
      'shiftLockAspectRatio',
      'spacePan',
      'altPickColor',
    ]);

    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'rect', shapeType: 'oval' }))).toEqual([
      'modErase',
      'shiftLockAspectRatio',
      'spacePan',
      'altPickColor',
    ]);
  });

  it('shows active rect and oval drag hints', () => {
    expect(
      getCanvasToolModifierHintIds(buildArgs({ tool: 'rect', shapeType: 'rect', isPrimaryPointerDown: true }))
    ).toEqual(['modErase', 'shiftLockAspectRatio', 'altScaleFromCenter', 'spaceMoveShape']);

    expect(
      getCanvasToolModifierHintIds(buildArgs({ tool: 'rect', shapeType: 'oval', isPrimaryPointerDown: true }))
    ).toEqual(['modErase', 'shiftLockAspectRatio', 'altScaleFromCenter', 'spaceMoveShape']);
  });

  it('shows polygon shape hints', () => {
    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'rect', shapeType: 'polygon' }))).toEqual([
      'modErase',
      'shiftSnap45Degrees',
      'spacePan',
      'altPickColor',
    ]);
  });

  it('omits alt color-picker hint during an active freehand stroke', () => {
    expect(getCanvasToolModifierHintIds(buildArgs({ tool: 'rect', shapeType: 'freehand' }))).toEqual([
      'modErase',
      'spacePan',
      'altPickColor',
    ]);

    expect(
      getCanvasToolModifierHintIds(buildArgs({ tool: 'rect', shapeType: 'freehand', isPrimaryPointerDown: true }))
    ).toEqual(['modErase', 'spacePan']);
  });
});
