import type { FillRegionParams } from '@workbench/canvas-engine/render/fill/regionMorphology';

import { createLineArtSample, LINE_ART } from '@workbench/canvas-engine/render/fill/fillImages.testing';
import { buildFillLevelMap } from '@workbench/canvas-engine/render/fill/fillLevelMap';
import {
  createFloodFillService,
  createFloodFillWorkerExecutor,
  type FloodFillService,
} from '@workbench/canvas-engine/render/fill/floodFillService';
import { computeFillRegion } from '@workbench/canvas-engine/render/fill/regionMorphology';
import { afterEach, describe, expect, it } from 'vitest';

const services: FloodFillService[] = [];

const createService = () => {
  let workers = 0;
  const service = createFloodFillService({
    createExecutor: () => {
      workers++;
      return createFloodFillWorkerExecutor();
    },
  });
  services.push(service);
  return { service, workers: () => workers };
};

afterEach(() => {
  for (const service of services.splice(0)) {
    service.dispose();
  }
});

describe('flood fill worker', () => {
  it('answers every region of a press exactly as the kernels do on the main thread', async () => {
    const { service, workers } = createService();
    const { seed } = LINE_ART.narrow;
    const map = buildFillLevelMap(createLineArtSample(), seed.x, seed.y, true);
    const first: FillRegionParams = { antiAlias: true, closeGaps: 0, expand: 0, tolerance: 20 };
    const later: FillRegionParams[] = [
      { antiAlias: false, closeGaps: 3, expand: 0, tolerance: 20 },
      { antiAlias: true, closeGaps: 3, expand: 2, tolerance: 20 },
    ];

    const sample = createLineArtSample();
    const begun = await service.begin(1, sample, seed, { contiguous: true, ...first });
    // Zero copy: the sample moved to the worker.
    expect(sample.data.byteLength).toBe(0);
    expect(begun).toEqual({ region: computeFillRegion(map, first), sequence: 0, status: 'ok' });
    for (const [index, params] of later.entries()) {
      expect(await service.region(1, params)).toEqual({
        region: computeFillRegion(map, params),
        sequence: index + 1,
        status: 'ok',
      });
    }

    service.end(1);
    expect((await service.region(1, first)).status).toBe('ended');
    expect(workers()).toBe(1);
  });

  it('reports a failed press as an error and keeps the same worker for the next one', async () => {
    const { service, workers } = createService();
    const params: FillRegionParams = { antiAlias: false, closeGaps: 0, expand: 0, tolerance: 50 };

    const outside = await service.begin(1, createLineArtSample(), { x: 999, y: 0 }, { contiguous: true, ...params });
    expect(outside.status).toBe('error');
    expect((await service.region(1, params)).status).toBe('error');
    service.end(1);

    const next = await service.begin(2, createLineArtSample(), { x: 0, y: 0 }, { contiguous: false, ...params });
    expect(next.status).toBe('ok');
    expect(workers()).toBe(1);
  });
});
