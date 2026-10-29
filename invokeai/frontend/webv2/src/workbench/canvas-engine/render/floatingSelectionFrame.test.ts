import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { FloatingSelection } from '@workbench/canvas-engine/selection/floatingSelection';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import { floatingSelectionFrame } from './floatingSelectionFrame';

const pixels = { surface: { id: 'raw' } as unknown as RasterSurface };

const float = (overrides: Partial<FloatingSelection> = {}): FloatingSelection =>
  ({
    display: null,
    layerId: 'layer-1',
    pixels: { rect: { height: 4, width: 4, x: 1, y: 2 }, surface: pixels.surface },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    ...overrides,
  }) as unknown as FloatingSelection;

const documentOf = (layerIds: readonly string[]): CanvasDocumentContractV3 =>
  ({
    stacks: stacksFrom(
      layerIds.map(
        (id) =>
          ({
            id,
            transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 },
            type: 'raster',
          }) as unknown as CanvasLayerContract
      )
    ),
  }) as unknown as CanvasDocumentContractV3;

describe('floatingSelectionFrame', () => {
  it('describes the composite and the ants together', () => {
    const frame = floatingSelectionFrame(float(), documentOf(['layer-1']));
    expect(frame).toMatchObject({
      composite: { layerId: 'layer-1', rect: { height: 4, width: 4, x: 1, y: 2 } },
    });
    expect(frame?.ants).toBeTruthy();
  });

  it('hands the compositor the same matrix the ants ride through', () => {
    const frame = floatingSelectionFrame(float(), documentOf(['layer-1']));
    // The ants matrix is the composite matrix projected into document space, so
    // a divergence here means the outline would drift off the drawn pixels.
    expect(frame?.composite.matrix).toBeTruthy();
    expect(frame?.ants).not.toBe(frame?.composite.matrix);
  });

  it.each([
    ['there is nothing floating', null, documentOf(['layer-1'])],
    ['there is no document', float(), null],
    ['the layer it was cut from is gone', float(), documentOf(['other'])],
  ])('reports nothing when %s', (_label, floating, doc) => {
    expect(floatingSelectionFrame(floating, doc)).toBeNull();
  });

  it('prefers the display copy so an effect layer draws what the user sees', () => {
    const display = { id: 'display' } as unknown as RasterSurface;
    const frame = floatingSelectionFrame(float({ display }), documentOf(['layer-1']));
    expect(frame?.composite.surface).toBe(display);
  });

  it('falls back to the raw pixels when the layer has no effect', () => {
    const frame = floatingSelectionFrame(float(), documentOf(['layer-1']));
    expect(frame?.composite.surface).toBe(pixels.surface);
  });

  it('reports nothing when the layer transform cannot be inverted', () => {
    const collapsed = {
      stacks: stacksFrom([
        {
          id: 'layer-1',
          transform: { rotation: 0, scaleX: 0, scaleY: 0, x: 0, y: 0 },
          type: 'raster',
        } as unknown as CanvasLayerContract,
      ]),
    } as unknown as CanvasDocumentContractV3;
    expect(floatingSelectionFrame(float(), collapsed)).toBeNull();
  });
});
