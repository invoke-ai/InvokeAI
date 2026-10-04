import { getFocusedRegion } from 'common/hooks/focus';
import type { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { atom } from 'nanostores';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { CanvasMoveToolModule } from './CanvasMoveToolModule';

vi.mock('common/hooks/focus', () => ({ getFocusedRegion: vi.fn(() => 'canvas') }));
vi.mock('features/controlLayers/konva/util', () => ({ getPrefixedId: (prefix: string) => prefix }));

const setup = () => {
  const entity = {
    $isDisabled: atom(false),
    $isEmpty: atom(false),
    $isLocked: atom(false),
    $isEntityTypeHidden: atom(false),
    transformer: {
      $isProcessing: atom(false),
      getIsTransformingVectorPath: vi.fn(() => true),
      nudgeBy: vi.fn(),
    },
  };
  const manager = {
    buildPath: () => [],
    buildLogger: () => ({ debug: vi.fn() }),
    $isBusy: atom(true),
    stateApi: {
      getSelectedEntityAdapter: vi.fn(() => entity),
      $transformingAdapter: atom<unknown>(entity),
    },
  };
  const parent = {
    manager,
    $tool: atom('rect'),
    tools: { path: { hasActiveEditSession: vi.fn(() => true) } },
  };
  return { entity, manager, parent, tool: new CanvasMoveToolModule(parent as unknown as CanvasToolModule) };
};

describe('path transform keyboard nudging', () => {
  beforeEach(() => vi.mocked(getFocusedRegion).mockReturnValue('canvas'));

  it.each([
    ['ArrowLeft', { x: -1, y: 0 }],
    ['ArrowRight', { x: 1, y: 0 }],
    ['ArrowUp', { x: 0, y: -1 }],
    ['ArrowDown', { x: 0, y: 1 }],
  ] as const)('routes %s to the active path transform, including repeated nudges', (key, offset) => {
    const { tool, entity } = setup();
    tool.nudge(key);
    tool.nudge(key);
    expect(entity.transformer.nudgeBy).toHaveBeenCalledTimes(2);
    expect(entity.transformer.nudgeBy).toHaveBeenLastCalledWith(offset);
  });

  it.each(['$isDisabled', '$isEmpty', '$isLocked', '$isEntityTypeHidden'] as const)(
    'does not nudge when %s is true',
    (flag) => {
      const { tool, entity } = setup();
      entity[flag].set(true);
      tool.nudge('ArrowRight');
      expect(entity.transformer.nudgeBy).not.toHaveBeenCalled();
    }
  );

  it('does not nudge outside the canvas or while applying the transform', () => {
    const { tool, entity } = setup();
    vi.mocked(getFocusedRegion).mockReturnValue(null);
    tool.nudge('ArrowRight');
    vi.mocked(getFocusedRegion).mockReturnValue('canvas');
    entity.transformer.$isProcessing.set(true);
    tool.nudge('ArrowRight');
    expect(entity.transformer.nudgeBy).not.toHaveBeenCalled();
  });

  it('does not move a different selected layer during path transformation', () => {
    const { tool, entity, manager } = setup();
    manager.stateApi.$transformingAdapter.set({});
    tool.nudge('ArrowRight');
    expect(entity.transformer.nudgeBy).not.toHaveBeenCalled();
  });

  it('does not move the whole layer in ordinary Edit mode', () => {
    const { tool, entity, manager, parent } = setup();
    entity.transformer.getIsTransformingVectorPath.mockReturnValue(false);
    manager.stateApi.$transformingAdapter.set(null);
    manager.$isBusy.set(false);
    parent.$tool.set('move');
    tool.nudge('ArrowRight');
    expect(entity.transformer.nudgeBy).not.toHaveBeenCalled();
  });

  it('preserves ordinary Move tool nudging outside Edit mode', () => {
    const { tool, entity, manager, parent } = setup();
    parent.tools.path.hasActiveEditSession.mockReturnValue(false);
    entity.transformer.getIsTransformingVectorPath.mockReturnValue(false);
    manager.stateApi.$transformingAdapter.set(null);
    manager.$isBusy.set(false);
    tool.nudge('ArrowRight');
    expect(entity.transformer.nudgeBy).not.toHaveBeenCalled();
    parent.$tool.set('move');
    tool.nudge('ArrowRight');
    expect(entity.transformer.nudgeBy).toHaveBeenCalledWith({ x: 1, y: 0 });
    manager.$isBusy.set(true);
    tool.nudge('ArrowRight');
    expect(entity.transformer.nudgeBy).toHaveBeenCalledOnce();
  });
});
