// SPDX-License-Identifier: Apache-2.0
import type { FloodFillExecutor } from '@workbench/canvas-engine/render/fill/floodFillService';

import { createFloodFillPresses } from '@workbench/canvas-engine/render/fill/floodFillPresses';

/**
 * Runs the worker's press state on the calling thread, each job and release after the previous one settles, so Node
 * tests keep the worker's asynchronous boundary and message order.
 */
export const createInlineFloodFillExecutor = (): FloodFillExecutor => {
  const presses = createFloodFillPresses();
  let tail: Promise<unknown> = Promise.resolve();
  return {
    dispose: () => undefined,
    release: (pressId) => {
      tail = tail.then(() => presses.release(pressId));
    },
    run: (job) => {
      const result = tail.then(() => presses.run(job));
      tail = result;
      return result;
    },
  };
};
