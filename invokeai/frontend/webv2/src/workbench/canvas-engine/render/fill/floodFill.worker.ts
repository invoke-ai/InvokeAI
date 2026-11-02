// SPDX-License-Identifier: Apache-2.0
/** Module worker behind `floodFillService`: runs one job per message and transfers the coverage back. */

import type {
  FloodFillWorkerRequest,
  FloodFillWorkerResponse,
} from '@workbench/canvas-engine/render/fill/floodFillService';

import { createFloodFillPresses } from '@workbench/canvas-engine/render/fill/floodFillPresses';

const presses = createFloodFillPresses();

self.addEventListener('message', (event: MessageEvent<FloodFillWorkerRequest>) => {
  const request = event.data;
  if (request.kind === 'release') {
    presses.release(request.pressId);
    return;
  }
  const response: FloodFillWorkerResponse = { ...presses.run(request), id: request.id };
  const coverage = response.status === 'ok' ? response.region.coverage : null;
  self.postMessage(response, { transfer: coverage ? [coverage.buffer] : [] });
});
