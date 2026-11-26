import type { OverlayState } from './render/overlayRenderer';
import type { RasterSurface } from './render/raster';

import { VectorEditingController } from './controllers/vectorEditingController';
import { traceVectorLayer } from './controllers/vectorTraceController';
import { createVectorLayer, nextVectorLayerName } from './document/vectorLayerFactories';
import { drawVectorEditOverlay } from './render/vectorEditOverlay';
import { drawVectorGuides } from './render/vectorGuides';
import { createVectorShapeTool } from './tools/vectorShapeTool';

/** Vector controllers, tools and rendering share the engine lifetime. */
export const createVectorRuntime = (options: ConstructorParameters<typeof VectorEditingController>[0]) => ({
  editor: new VectorEditingController(options),
  shape: createVectorShapeTool(),
  createLayer: createVectorLayer,
  nextLayerName: nextVectorLayerName,
  trace: traceVectorLayer,
  render: (ctx: RasterSurface['ctx'], state: OverlayState): void => {
    drawVectorGuides(ctx, state.view, state.vectorGuides ?? [], state.vectorShapePreview ?? null);
    drawVectorEditOverlay(ctx, state.view, state.vectorEdit ?? null);
  },
});

export type VectorRuntime = ReturnType<typeof createVectorRuntime>;
