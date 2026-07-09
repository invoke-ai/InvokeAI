import { describe, expect, it } from 'vitest';

import type { CanvasLayerContextMenuTarget } from './LayerContextMenu';

import { createLayerMenuTargetFromContextEvent, resolveMenuTargetForRender } from './layerMenuState';

const target = (layerId: string): CanvasLayerContextMenuTarget => ({ layerId, x: 10, y: 20 });

describe('resolveMenuTargetForRender', () => {
  it('renders the live target while the menu is open', () => {
    const live = target('a');
    expect(resolveMenuTargetForRender(live, null)).toBe(live);
  });

  it('renders nothing with no live target and no pending rename', () => {
    expect(resolveMenuTargetForRender(null, null)).toBeNull();
  });

  it('falls back to the captured rename target after the menu closes (F1: dialog survives)', () => {
    const renaming = target('a');
    expect(resolveMenuTargetForRender(null, renaming)).toBe(renaming);
  });

  it('prefers the live target over the rename target when both exist', () => {
    const live = target('b');
    expect(resolveMenuTargetForRender(live, target('a'))).toBe(live);
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
