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

/** The engine's input wiring over a real surface: focusable container, overlay canvas, window key listeners. */
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
  const tools = Object.fromEntries(
    (['brush', 'view', 'transform'] as const).map((id): [ToolId, Tool] => [
      id,
      {
        id,
        onKeyCommand: (_ctx, command) => events.push(`${id}:${command}`),
        onPointerCancel: () => events.push(`${id}:cancel`),
        onPointerDown: () => events.push(`${id}:down`),
        onPointerMove: () => events.push(`${id}:move`),
        onPointerUp: () => events.push(`${id}:up`),
      },
    ])
  ) as Record<string, Tool>;
  const ctx = {} as ToolContext;
  const interaction = new InteractionController({
    cancelGesture: () => pipeline.cancelGestureForToolSwitch(),
    getTool: (id) => tools[id],
    getToolContext: () => ctx,
    initialToolId: 'brush',
    invalidateOverlay: () => undefined,
    isLocked: () => false,
    publishActiveTool: () => undefined,
    stepBrushSize: () => undefined,
    updateCursor: () => undefined,
  });
  const pipeline = createPointerPipeline({
    getActiveTool: () => interaction.getActiveTool(),
    getActiveToolId: () => interaction.getActiveToolId(),
    getInputElement: () => canvas,
    getToolContext: () => ctx,
    hasTool: (id) => id in tools,
    setTool: (id, options) => interaction.setTool(id, options),
    updateCursor: () => undefined,
    viewport: createViewport(),
  });
  // Registered after the pipeline's listener, so it reads the event after the pipeline handled it.
  const observed: { type: string; defaultPrevented: boolean }[] = [];
  const observe = (event: KeyboardEvent): void => {
    observed.push({ defaultPrevented: event.defaultPrevented, type: event.type });
  };
  globalThis.addEventListener('keydown', pipeline.onKeyDown);
  globalThis.addEventListener('keyup', pipeline.onKeyUp);
  globalThis.addEventListener('keydown', observe);
  globalThis.addEventListener('keyup', observe);
  cleanups.push(() => {
    globalThis.removeEventListener('keydown', pipeline.onKeyDown);
    globalThis.removeEventListener('keyup', pipeline.onKeyUp);
    globalThis.removeEventListener('keydown', observe);
    globalThis.removeEventListener('keyup', observe);
    pipeline.reset();
    surface.remove();
    button.remove();
  });
  const pointer = (type: string, buttons: number): PointerEvent =>
    new PointerEvent(type, { bubbles: true, button: 0, buttons, pointerId: 1 });
  return { button, canvas, clicks: () => clicks, events, interaction, observed, pipeline, pointer, surface };
};

describe('pointer pipeline in Chromium: keyboard ownership', () => {
  it('leaves Enter and Space to a focused toolbar control', async () => {
    const h = createInputHarness();
    h.interaction.setTool('transform');
    h.pipeline.onPointerEnter();
    h.button.focus();

    await userEvent.keyboard('{Enter}');
    await userEvent.keyboard('{Control>}{Enter}{/Control}');
    await userEvent.keyboard(' ');
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

  it('restores a held tool when Space is released after focus moved, without consuming the release', async () => {
    const h = createInputHarness();
    h.pipeline.onPointerEnter();
    h.surface.focus();

    await userEvent.keyboard('{Space>}');
    expect(h.interaction.getActiveToolId()).toBe('view');
    h.pipeline.onPointerLeave();
    h.button.focus();
    await userEvent.keyboard('{/Space}');

    expect(h.interaction.getActiveToolId()).toBe('brush');
    expect(h.observed.at(-1)).toEqual({ defaultPrevented: false, type: 'keyup' });
    expect(h.clicks()).toBe(0);
  });
});

describe('pointer pipeline in Chromium: gesture ownership across tool switches', () => {
  it('cancels a drag through the outgoing tool on a genuine switch and never routes its tail to the new tool', () => {
    const h = createInputHarness();
    h.pipeline.onPointerDown(h.pointer('pointerdown', 1));
    h.interaction.setTool('transform');

    expect(h.pipeline.isGestureActive()).toBe(false);
    h.pipeline.onPointerMove(h.pointer('pointermove', 1));
    h.pipeline.onPointerUp(h.pointer('pointerup', 0));
    expect(h.events).toEqual(['brush:down', 'brush:cancel']);
  });

  it('reset cancels the temporary tool that owns the gesture, then restores the held tool', () => {
    const h = createInputHarness();
    h.pipeline.onPointerEnter();
    h.surface.focus();
    h.surface.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, code: 'Space', key: ' ' }));
    h.pipeline.onPointerDown(h.pointer('pointerdown', 1));

    h.pipeline.reset();
    expect(h.events).toEqual(['view:down', 'view:cancel']);
    expect(h.interaction.getActiveToolId()).toBe('brush');
  });
});
