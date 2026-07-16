/**
 * The brush tool: paints variable-width, pressure-sensitive strokes in the brush
 * color into the target paint layer's cache surface. It is a thin binding over
 * {@link createPaintTool} that sources its size/color/opacity/pressure from the
 * engine's `brushOptions` store; all gesture and pixel logic lives in the shared
 * paint tool and {@link createStrokeSession}.
 *
 * Each engine builds its own instance so gesture state is per-engine. Zero React,
 * zero import-time side effects.
 */

import { PRESSURE_THINNING } from '@workbench/canvas-engine/tools/paintConstants';
import { createPaintTool } from '@workbench/canvas-engine/tools/paintTool';

import type { Tool } from './tool';

/** Creates a fresh brush tool with its own gesture state. */
export const createBrushTool = (): Tool =>
  createPaintTool({
    color: (ctx) => ctx.stores.brushOptions.get().color,
    composite: 'source-over',
    id: 'brush',
    opacity: (ctx) => ctx.stores.brushOptions.get().opacity,
    size: (ctx) => ctx.stores.brushOptions.get().size,
    thinning: (ctx) => (ctx.stores.brushOptions.get().pressureSensitivity ? PRESSURE_THINNING : 0),
  });
