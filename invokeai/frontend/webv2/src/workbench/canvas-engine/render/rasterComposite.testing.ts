import type { CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

/**
 * A noisy text layer the browser measures wider than its estimate. Estimated 240×100 (0.6 em per glyph); measured
 * 400×100. At scale 0.02 the 8×4 output samples the whole layer-local region, so noise adjusts 400×100 pixels where
 * the composite plan expected 240×100.
 */
export const MEASURED_NOISY_TEXT = {
  bbox: { height: 4, width: 8, x: 0, y: 0 } satisfies Rect,
  /** Scratch pixels beyond the plan that rendering must reserve. */
  excessPixels: 400 * 100 - 240 * 100,
  layer: {
    adjustments: [
      {
        amount: 0.2,
        distribution: 'gaussian',
        grain: 'monochrome',
        id: 'grain',
        isEnabled: true,
        response: 'flat',
        seed: 5,
        size: 1,
        type: 'noise',
      },
    ],
    blendMode: 'normal',
    id: 'text',
    isEnabled: true,
    isLocked: false,
    name: 'text',
    opacity: 1,
    source: {
      align: 'left',
      color: '#fff',
      content: 'WWWW',
      fontFamily: 'sans-serif',
      fontSize: 100,
      fontWeight: 400,
      lineHeight: 1,
      type: 'text',
    },
    transform: { rotation: 0, scaleX: 0.02, scaleY: 0.02, x: 0, y: 0 },
    type: 'raster',
  } satisfies CanvasRasterLayerContractV2,
  measured: { height: 100, width: 400, x: 0, y: 0 } satisfies Rect,
  /** The scratch the plan reserves from the estimate. */
  plannedScratchPixels: 240 * 100,
} as const;
