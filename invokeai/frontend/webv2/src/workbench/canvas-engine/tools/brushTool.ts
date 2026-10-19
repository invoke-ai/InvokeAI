import { PRESSURE_THINNING } from '@workbench/canvas-engine/tools/paintConstants';
import { createPaintTool } from '@workbench/canvas-engine/tools/paintTool';

import type { Tool } from './tool';

/** Creates a fresh brush tool with its own gesture state. */
export const createBrushTool = (): Tool =>
  createPaintTool({
    color: (ctx) => ctx.stores.brushOptions.get().color,
    composite: 'source-over',
    hardness: (ctx) => ctx.stores.brushOptions.get().hardness,
    id: 'brush',
    opacity: (ctx) => ctx.stores.brushOptions.get().opacity,
    size: (ctx) => ctx.stores.brushOptions.get().size,
    pressureOpacity: (ctx) => ctx.stores.brushOptions.get().pressureAffectsOpacity,
    thinning: (ctx) => (ctx.stores.brushOptions.get().pressureAffectsWidth ? PRESSURE_THINNING : 0),
  });
