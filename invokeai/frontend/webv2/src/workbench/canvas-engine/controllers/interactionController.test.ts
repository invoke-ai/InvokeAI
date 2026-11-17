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
  const order: string[] = [];
  const beforeSwitch = vi.fn(() => order.push('beforeSwitch'));
  const cancelGesture = vi.fn(() => order.push('cancelGesture'));
  vi.mocked(view.onDeactivate!).mockImplementation(() => order.push('view:deactivate'));
  const stepBrushSize = vi.fn();
  const controller = new InteractionController({
    beforeSwitch,
    cancelGesture,
    getTool: (id) => tools[id as keyof typeof tools],
    getToolContext: () => ({}) as never,
    invalidateOverlay,
    publishActiveTool,
    stepBrushSize,
    updateCursor,
  });
  return { cancelGesture, controller, order, stepBrushSize };
};

describe('InteractionController', () => {
  it('cancels an in-flight gesture through the outgoing tool before a genuine switch only', () => {
    const h = createHarness();
    h.controller.setTool('brush', { temporary: true });
    expect(h.cancelGesture).not.toHaveBeenCalled();

    h.controller.setTool('view');
    h.order.length = 0;
    h.controller.setTool('brush');
    expect(h.order).toEqual(['cancelGesture', 'beforeSwitch', 'view:deactivate']);
  });

  it('ignores commands after disposal', () => {
    const h = createHarness();
    h.controller.tools.stepBrushSize(1);
    expect(h.stepBrushSize).toHaveBeenCalledWith(1);
    h.controller.dispose();
    h.controller.dispose();
    h.controller.tools.stepBrushSize(-1);
    h.controller.setTool('brush');
    expect(h.stepBrushSize).toHaveBeenCalledOnce();
    expect(h.controller.getActiveToolId()).toBe('view');
  });
});
