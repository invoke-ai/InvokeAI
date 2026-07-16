/**
 * The eraser tool: clears pixels along a stroke by compositing the same freehand
 * shape as the brush with `destination-out` into the target paint layer's cache.
 * A thin binding over {@link createPaintTool} sourcing size/opacity from the
 * engine's `eraserOptions` store; the eraser is pressure-insensitive (thinning 0)
 * and has no color (the shape alone drives the erase).
 *
 * Each engine builds its own instance so gesture state is per-engine. Zero React,
 * zero import-time side effects.
 */

import { createPaintTool } from '@workbench/canvas-engine/tools/paintTool';

import type { Tool } from './tool';

/** Creates a fresh eraser tool with its own gesture state. */
export const createEraserTool = (): Tool =>
  createPaintTool({
    color: () => '#000000',
    composite: 'destination-out',
    id: 'eraser',
    opacity: (ctx) => ctx.stores.eraserOptions.get().opacity,
    size: (ctx) => ctx.stores.eraserOptions.get().size,
    thinning: () => 0,
  });
