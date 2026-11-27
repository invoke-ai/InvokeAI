import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { ToolId } from '@workbench/canvas-engine/types';

import { InteractionController } from '@workbench/canvas-engine/controllers/interactionController';
import { createPointerPipeline } from '@workbench/canvas-engine/input/pointerPipeline';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { afterEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

const cleanups: (() => void)[] = [];

afterEach(() => {
  for (const cleanup of cleanups.splice(0).reverse()) {
    cleanup();
  }
});

const createInputHarness = () => {
  const surface = document.createElement('div');
  surface.tabIndex = -1;
  const canvas = document.createElement('canvas');
  // Synthetic pointer ids are never active, so real capture would throw.
  canvas.setPointerCapture = () => undefined;
  canvas.releasePointerCapture = () => undefined;
  surface.append(canvas);
  const button = document.createElement('button');
  button.textContent = 'Toolbar action';
  let clicks = 0;
  button.addEventListener('click', () => {
    clicks += 1;
  });
  document.body.append(surface, button);

  const events: string[] = [];
  const deactivations: string[] = [];
  const tools = Object.fromEntries(
    (['brush', 'view', 'transform', 'bbox', 'eraser', 'colorPicker'] as const).map((id): [ToolId, Tool] => [
      id,
      {
        id,
        onDeactivate: (_ctx, options) => {
          if (!options?.temporary) {
            deactivations.push(id);
          }
        },
        onKeyCommand: (_ctx, command) => events.push(`${id}:${command}`) > 0,
        onPointerCancel: () => events.push(`${id}:cancel`),
        onPointerDown: () => events.push(`${id}:down`),
        onPointerMove: () => events.push(`${id}:move`),
        onPointerUp: () => events.push(`${id}:up`),
      },
    ])
  ) as Record<string, Tool>;
  const ctx = {} as ToolContext;
  const interaction = new InteractionController({
    cancelGesture: () => pipeline.endForToolSwitch(),
    getTool: (id) => tools[id],
    getToolContext: () => ctx,
    initialToolId: 'brush',
    invalidateOverlay: () => undefined,
    publishActiveTool: () => undefined,
    stepBrushSize: () => undefined,
    updateCursor: () => undefined,
  });
  const pipeline = createPointerPipeline({
    getActiveTool: () => interaction.getActiveTool(),
    getActiveToolId: () => interaction.getActiveToolId(),
    getInputElement: () => canvas,
    getKeyboardRoot: () => surface,
    getToolContext: () => ctx,
    handleEscape: () => events.push('escape'),
    hasTool: (id) => id in tools,
    isReplaying: () => false,
    setTool: (id, options) => interaction.setTool(id, options),
    updateCursor: () => undefined,
    viewport: createViewport(),
  });
  const observed: { type: string; defaultPrevented: boolean }[] = [];
  const observe = (event: KeyboardEvent): void => {
    observed.push({ defaultPrevented: event.defaultPrevented, type: event.type });
  };
  globalThis.addEventListener('keydown', pipeline.onKeyDown);
  globalThis.addEventListener('keyup', pipeline.onKeyUp);
  globalThis.addEventListener('focusin', pipeline.onFocusIn);
  globalThis.addEventListener('keydown', observe);
  globalThis.addEventListener('keyup', observe);
  cleanups.push(() => {
    globalThis.removeEventListener('keydown', pipeline.onKeyDown);
    globalThis.removeEventListener('keyup', pipeline.onKeyUp);
    globalThis.removeEventListener('focusin', pipeline.onFocusIn);
    globalThis.removeEventListener('keydown', observe);
    globalThis.removeEventListener('keyup', observe);
    pipeline.reset();
    surface.remove();
    button.remove();
  });
  const pointer = (type: string, buttons: number): PointerEvent =>
    new PointerEvent(type, { bubbles: true, button: 0, buttons, pointerId: 1 });
  return {
    button,
    canvas,
    clicks: () => clicks,
    deactivations,
    events,
    interaction,
    observed,
    pipeline,
    pointer,
    surface,
  };
};

describe('pointer pipeline in Chromium: keyboard ownership', () => {
  it('leaves Enter, Space, C and Escape to a control reached by keyboard', async () => {
    const h = createInputHarness();
    h.interaction.setTool('transform');
    h.pipeline.onPointerEnter();
    h.surface.focus();
    await userEvent.tab();
    expect(document.activeElement).toBe(h.button);

    await userEvent.keyboard('{Enter}');
    await userEvent.keyboard('{Control>}{Enter}{/Control}');
    await userEvent.keyboard(' ');
    await userEvent.keyboard('c');
    await userEvent.keyboard('{Escape}');
    expect(h.clicks()).toBe(2);
    expect(h.events).toEqual([]);
    expect(h.interaction.getActiveToolId()).toBe('transform');
    expect(h.observed.filter((event) => event.defaultPrevented)).toEqual([]);
  });

  it('applies on Enter only when the focused surface owns an unmodified, unprevented key', async () => {
    const h = createInputHarness();
    h.interaction.setTool('transform');
    h.surface.focus();

    await userEvent.keyboard('{Control>}{Enter}{/Control}');
    await userEvent.keyboard('{Shift>}{Enter}{/Shift}');
    const prevented = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Enter' });
    prevented.preventDefault();
    h.surface.dispatchEvent(prevented);
    expect(h.events).toEqual([]);

    await userEvent.keyboard('{Enter}');
    expect(h.events).toEqual(['transform:apply']);
  });
});

describe('pointer pipeline in Chromium: tool switches during a hold', () => {
  it('choosing the held tool again only ends the hold', async () => {
    const h = createInputHarness();
    h.pipeline.onPointerEnter();
    h.surface.focus();

    await userEvent.keyboard('{Space>}');
    h.interaction.setTool('brush');
    expect(h.interaction.getActiveToolId()).toBe('brush');
    expect(h.deactivations).toEqual([]);
    await userEvent.keyboard('{/Space}');
    expect(h.interaction.getActiveToolId()).toBe('brush');
  });
});

describe('pointer pipeline in Chromium: Alt eyedropper', () => {
  it('prevents a real Alt tap on the focused canvas, keydown and keyup alike', async () => {
    const h = createInputHarness();
    h.pipeline.onPointerEnter();
    h.surface.focus();

    await userEvent.keyboard('{Alt}');
    expect(h.observed).toEqual([
      { defaultPrevented: true, type: 'keydown' },
      { defaultPrevented: true, type: 'keyup' },
    ]);
    expect(h.interaction.getActiveToolId()).toBe('brush');
  });
});
