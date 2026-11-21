import type { ToolContext } from '@workbench/canvas-engine/tools/tool';

import { createPointerPipeline } from '@workbench/canvas-engine/input/pointerPipeline';
import { createWheelInput } from '@workbench/canvas-engine/input/wheel';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { afterEach, describe, expect, it } from 'vitest';

import { RenderController } from './renderController';

const cleanups: (() => void)[] = [];

afterEach(() => {
  for (const cleanup of cleanups.splice(0).reverse()) {
    cleanup();
  }
});

/** A real pipeline bound through `attach`, with a control that stops every key event it receives. */
const createAttachedInput = () => {
  const surface = document.createElement('div');
  const screen = document.createElement('canvas');
  const overlay = document.createElement('canvas');
  const field = document.createElement('input');
  const stop = (event: Event): void => event.stopPropagation();
  field.addEventListener('keydown', stop);
  field.addEventListener('keyup', stop);
  surface.append(screen, overlay);
  document.body.append(surface, field);

  const ctx = {} as ToolContext;
  const viewport = createViewport();
  const pipeline = createPointerPipeline({
    getActiveTool: () => undefined,
    getActiveToolId: () => 'view',
    getInputElement: () => overlay,
    getKeyboardRoot: () => surface,
    getToolContext: () => ctx,
    hasTool: () => false,
    isReplaying: () => false,
    setTool: () => undefined,
    updateCursor: () => undefined,
    viewport,
  });
  const wheel = createWheelInput({
    getActiveTool: () => undefined,
    getControlKeyState: () => pipeline.controlKeyState(),
    getInputElement: () => overlay,
    getInvertBrushSizeScroll: () => false,
    getScrollInput: () => 'auto',
    invalidate: () => undefined,
    isGestureActive: () => pipeline.isGestureActive(),
    stepActiveBrushSize: () => undefined,
    trackpadDetected: { get: () => false, set: () => undefined },
    viewport,
  });
  const controller = new RenderController({
    applyCursor: () => undefined,
    getInputHandlers: () => ({ ...pipeline, ...wheel }),
    isEngineDisposed: () => false,
    onPageHide: () => undefined,
    onVisibilityChange: () => undefined,
    onWindowBlur: () => pipeline.reset(),
    render: () => undefined,
    setViewportReady: () => undefined,
    updateAnimation: () => undefined,
    updateCursor: () => undefined,
  });
  controller.attach(screen, overlay, surface);
  cleanups.push(() => {
    controller.dispose();
    surface.remove();
    field.remove();
  });
  const control = (type: 'keydown' | 'keyup') =>
    field.dispatchEvent(
      new KeyboardEvent(type, { bubbles: true, code: 'ControlLeft', ctrlKey: type === 'keydown', key: 'Control' })
    );
  const pinch = (type: 'gesturestart' | 'gesturechange', scale: number): Event => {
    const event = Object.assign(new Event(type, { cancelable: true }), { clientX: 0, clientY: 0, scale });
    overlay.dispatchEvent(event);
    return event;
  };
  return { control, controller, pinch, pipeline, viewport };
};

describe('RenderController in Chromium: physical Control key', () => {
  it('sees key events a control stops from bubbling, and forgets the key when the window regains focus', () => {
    const { control, pipeline } = createAttachedInput();
    control('keydown');
    expect(pipeline.controlKeyState()).toBe('down');
    control('keyup');
    expect(pipeline.controlKeyState()).toBe('up');

    globalThis.dispatchEvent(new FocusEvent('focus'));
    expect(pipeline.controlKeyState()).toBe('unknown');
  });

  it('stops tracking once detached', () => {
    const { control, controller, pipeline } = createAttachedInput();
    controller.detach();
    control('keydown');
    expect(pipeline.controlKeyState()).toBe('unknown');
  });
});

describe('RenderController in Chromium: Safari pinch gestures', () => {
  it('takes them over from the page zoom while attached, and lets go once detached', () => {
    const { controller, pinch, viewport } = createAttachedInput();
    expect(pinch('gesturestart', 1).defaultPrevented).toBe(true);
    expect(pinch('gesturechange', 2).defaultPrevented).toBe(true);
    expect(viewport.getZoom()).toBeCloseTo(2, 10);

    controller.detach();
    expect(pinch('gesturechange', 3).defaultPrevented).toBe(false);
    expect(viewport.getZoom()).toBeCloseTo(2, 10);
  });
});
