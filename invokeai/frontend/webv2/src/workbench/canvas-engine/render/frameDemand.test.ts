import type { CanvasDocumentContractV2, CanvasRasterLayerContractV2 } from '@workbench/types';

import { describe, expect, it } from 'vitest';

import { calculateActiveFrameLayerIds } from './frameDemand';

const layer = (id: string, x: number, y: number, width = 100, height = 100): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height, imageName: `${id}.png`, width }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y },
  type: 'raster',
});

const document = (layers: CanvasRasterLayerContractV2[]): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 1_000,
  layers,
  selectedLayerId: null,
  version: 2,
  width: 1_000,
});

describe('calculateActiveFrameLayerIds', () => {
  it('changes demand deterministically when panning reveals an offscreen layer', () => {
    const doc = document([layer('left', 0, 0), layer('right', 500, 0)]);

    expect(calculateActiveFrameLayerIds({ document: doc, viewport: { height: 200, width: 200, x: 0, y: 0 } })).toEqual(
      new Set(['left'])
    );
    expect(
      calculateActiveFrameLayerIds({ document: doc, viewport: { height: 200, width: 200, x: 450, y: 0 } })
    ).toEqual(new Set(['right']));
  });

  it('uses live cache bounds for unflushed paint pixels', () => {
    const paint = { ...layer('paint', 20, 20), source: { bitmap: null, type: 'paint' as const } };

    expect(
      calculateActiveFrameLayerIds({
        document: document([paint]),
        liveCacheRects: new Map([['paint', { height: 30, width: 40, x: 0, y: 0 }]]),
        viewport: { height: 100, width: 100, x: 0, y: 0 },
      })
    ).toEqual(new Set(['paint']));
  });

  it('limits demand to isolated layers', () => {
    const doc = document([layer('isolated', 0, 0), layer('covered', 0, 0)]);

    expect(
      calculateActiveFrameLayerIds({
        document: doc,
        isolationLayerIds: new Set(['isolated']),
        viewport: { height: 200, width: 200, x: 0, y: 0 },
      })
    ).toEqual(new Set(['isolated']));
  });

  it('uses a transient transform override before cache allocation', () => {
    const moving = layer('moving', 500, 0);

    expect(
      calculateActiveFrameLayerIds({
        document: document([moving]),
        transformOverrides: new Map([['moving', { x: 20, y: 20 }]]),
        viewport: { height: 200, width: 200, x: 0, y: 0 },
      })
    ).toEqual(new Set(['moving']));
  });

  it('uses transformed corners for rotated-layer demand', () => {
    const rotated = {
      ...layer('rotated', 250, 0),
      transform: { rotation: Math.PI / 2, scaleX: 1, scaleY: 1, x: 250, y: 0 },
    };

    expect(
      calculateActiveFrameLayerIds({
        document: document([rotated]),
        viewport: { height: 120, width: 120, x: 140, y: 0 },
      })
    ).toEqual(new Set(['rotated']));
  });
});
