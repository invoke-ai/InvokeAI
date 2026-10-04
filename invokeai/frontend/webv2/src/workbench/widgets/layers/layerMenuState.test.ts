import { describe, expect, it } from 'vitest';

import type { CanvasLayerContextMenuTarget } from './LayerContextMenu';

import {
  createLayerMenuTargetFromContextEvent,
  type LayerMenuDialogKind,
  resolveMenuTargetForRender,
} from './layerMenuState';

const target = (layerId: string): CanvasLayerContextMenuTarget => ({ layerId, x: 10, y: 20 });

describe('resolveMenuTargetForRender', () => {
  it('renders the live target while the menu is open', () => {
    const live = target('a');
    expect(resolveMenuTargetForRender(live, null)).toBe(live);
  });

  it('renders nothing with no live target and no pending dialog', () => {
    expect(resolveMenuTargetForRender(null, null)).toBeNull();
  });

  it.each(['rename', 'run-workflow'] as const)(
    'falls back to the captured %s target after the menu closes',
    (kind: LayerMenuDialogKind) => {
      const dialogTarget = target('a');
      expect(resolveMenuTargetForRender(null, { kind, target: dialogTarget })).toBe(dialogTarget);
    }
  );

  it('prefers the live target over a pending dialog target when both exist', () => {
    const live = target('b');
    expect(resolveMenuTargetForRender(live, { kind: 'rename', target: target('a') })).toBe(live);
  });

  it('clears the sticky target after dialog close', () => {
    const dialogState = { kind: 'rename', target: target('a') } as const;
    expect(resolveMenuTargetForRender(null, dialogState)).toBe(dialogState.target);
    expect(resolveMenuTargetForRender(null, null)).toBeNull();
  });
});

describe('createLayerMenuTargetFromContextEvent', () => {
  it('uses the pointer position and suppresses the browser context menu', () => {
    let didPreventDefault = false;
    let didStopPropagation = false;

    expect(
      createLayerMenuTargetFromContextEvent('layer-1', {
        clientX: 30,
        clientY: 40,
        preventDefault: () => {
          didPreventDefault = true;
        },
        stopPropagation: () => {
          didStopPropagation = true;
        },
      })
    ).toEqual({ layerId: 'layer-1', x: 30, y: 40 });
    expect(didPreventDefault).toBe(true);
    expect(didStopPropagation).toBe(true);
  });
});
