// SPDX-License-Identifier: Apache-2.0
import type { SpatialAdjustmentExecutor } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';

import { evaluateSpatialSteps } from '@workbench/canvas-engine/render/spatial/evaluateSpatialSteps';

/** Evaluates on the calling thread after a microtask, so Node tests keep the worker's asynchronous boundary. */
export const createInlineSpatialExecutor = (): SpatialAdjustmentExecutor => ({
  dispose: () => undefined,
  run: async ({ buffer, height, steps, width }) => {
    await Promise.resolve();
    try {
      return { buffer, status: evaluateSpatialSteps(new Uint8ClampedArray(buffer), width, height, steps).status };
    } catch (error) {
      return { buffer, message: error instanceof Error ? error.message : String(error), status: 'error' };
    }
  },
});
