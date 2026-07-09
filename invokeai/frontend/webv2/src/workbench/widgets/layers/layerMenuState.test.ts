import { describe, expect, it } from 'vitest';

import type { CanvasLayerContextMenuTarget } from './LayerContextMenu';

import { resolveMenuTargetForRender } from './layerMenuState';

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
