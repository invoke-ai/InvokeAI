/**
 * Eraser binds shared paint gestures to destination-out with eraser size/opacity, no color and pressure thinning
 * zero. Gesture state is per engine.
 */

import { createPaintTool } from '@workbench/canvas-engine/tools/paintTool';

import type { Tool } from './tool';

/** Creates a fresh eraser tool with its own gesture state. */
export const createEraserTool = (): Tool =>
  createPaintTool({
    color: () => '#000000',
    composite: 'destination-out',
    id: 'eraser',
    hardness: (ctx) => ctx.stores.eraserOptions.get().hardness,
    opacity: (ctx) => ctx.stores.eraserOptions.get().opacity,
    pressureOpacity: () => false,
    size: (ctx) => ctx.stores.eraserOptions.get().size,
    thinning: () => 0,
  });
