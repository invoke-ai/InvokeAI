import type { Tool } from '@workbench/canvas-engine/tools/tool';

import { describe, expect, it, vi } from 'vitest';

import { InteractionController } from './interactionController';

const createHarness = () => {
  const brush = { onActivate: vi.fn(), onDeactivate: vi.fn() } as unknown as Tool;
  const view = { onActivate: vi.fn(), onDeactivate: vi.fn() } as unknown as Tool;
  const tools = { brush, view };
  const publishActiveTool = vi.fn();
  const updateCursor = vi.fn();
  const invalidateOverlay = vi.fn();
  const beforeSwitch = vi.fn();
  const stepBrushSize = vi.fn();
  let locked = false;
  const controller = new InteractionController({
    beforeSwitch,
    getTool: (id) => tools[id as keyof typeof tools],
    getToolContext: () => ({}) as never,
    invalidateOverlay,
    isLocked: () => locked,
    publishActiveTool,
    stepBrushSize,
    updateCursor,
  });
  return {
    beforeSwitch,
    brush,
    controller,
    invalidateOverlay,
    lock: () => (locked = true),
    publishActiveTool,
    stepBrushSize,
    updateCursor,
    view,
  };
};

describe('InteractionController', () => {
  it('owns active-tool transitions and publishes their effects', () => {
    const h = createHarness();
    h.controller.tools.setTool('brush', { temporary: true });

    expect(h.beforeSwitch).toHaveBeenCalledWith('view', 'brush', { temporary: true });
    expect(h.view.onDeactivate).toHaveBeenCalledOnce();
    expect(h.brush.onActivate).toHaveBeenCalledOnce();
    expect(h.controller.getActiveToolId()).toBe('brush');
    expect(h.publishActiveTool).toHaveBeenCalledWith('brush');
    expect(h.updateCursor).toHaveBeenCalledOnce();
    expect(h.invalidateOverlay).toHaveBeenCalledOnce();
  });

  it('blocks non-view transitions while locked and ignores commands after disposal', () => {
    const h = createHarness();
    h.lock();
    h.controller.setTool('brush');
    expect(h.controller.getActiveToolId()).toBe('view');

    h.controller.tools.stepBrushSize(1);
    expect(h.stepBrushSize).toHaveBeenCalledWith(1);
    h.controller.dispose();
    h.controller.dispose();
    h.controller.tools.stepBrushSize(-1);
    expect(h.stepBrushSize).toHaveBeenCalledOnce();
  });
});
