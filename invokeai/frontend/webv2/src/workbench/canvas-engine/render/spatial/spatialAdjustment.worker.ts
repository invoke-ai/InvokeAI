// SPDX-License-Identifier: Apache-2.0
/** Module worker behind `spatialAdjustmentService`: evaluates one job per message and transfers the buffer back. */

import type {
  SpatialExecutorResult,
  SpatialWorkerRequest,
  SpatialWorkerResponse,
} from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';

import { evaluateSpatialSteps } from '@workbench/canvas-engine/render/spatial/evaluateSpatialSteps';

const evaluate = ({ buffer, height, steps, width }: SpatialWorkerRequest): SpatialExecutorResult => {
  try {
    return { buffer, status: evaluateSpatialSteps(new Uint8ClampedArray(buffer), width, height, steps).status };
  } catch (error) {
    return { buffer, message: error instanceof Error ? error.message : String(error), status: 'error' };
  }
};

self.addEventListener('message', (event: MessageEvent<SpatialWorkerRequest>) => {
  const response: SpatialWorkerResponse = { ...evaluate(event.data), id: event.data.id };
  self.postMessage(response, { transfer: [response.buffer] });
});
