// SPDX-License-Identifier: Apache-2.0
/** Worker-side fill state: each open press keeps its level map and region scratch until it is released. */

import type { FloodFillExecutorResult, FloodFillJob } from '@workbench/canvas-engine/render/fill/floodFillService';

import { buildFillLevelMap, type FillLevelMap } from '@workbench/canvas-engine/render/fill/fillLevelMap';
import {
  computeFillRegion,
  createRegionScratch,
  type RegionScratch,
} from '@workbench/canvas-engine/render/fill/regionMorphology';

export interface FloodFillPresses {
  run(job: FloodFillJob): FloodFillExecutorResult;
  release(pressId: number): void;
}

export const createFloodFillPresses = (): FloodFillPresses => {
  const presses = new Map<number, { readonly map: FillLevelMap; readonly scratch: RegionScratch }>();
  return {
    release: (pressId) => {
      presses.delete(pressId);
    },
    run: (job) => {
      try {
        let press = presses.get(job.pressId);
        if (job.kind === 'begin') {
          const sample = { data: new Uint8ClampedArray(job.buffer), height: job.height, width: job.width };
          press = {
            map: buildFillLevelMap(sample, job.seedX, job.seedY, job.contiguous),
            scratch: createRegionScratch(),
          };
          presses.set(job.pressId, press);
        } else if (!press) {
          return { message: `Fill press ${job.pressId} has no level map.`, status: 'error' };
        }
        return { region: computeFillRegion(press.map, job.params, press.scratch), status: 'ok' };
      } catch (error) {
        return { message: error instanceof Error ? error.message : String(error), status: 'error' };
      }
    },
  };
};
