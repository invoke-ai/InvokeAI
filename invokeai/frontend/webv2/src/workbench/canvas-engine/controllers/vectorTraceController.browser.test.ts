import { documentFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createVectorLayer } from '@workbench/canvas-engine/document/vectorLayerFactories';
import { anchorsToBezierPoints } from '@workbench/canvas-engine/math/bezierPath';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

import { createLayerOperationHarness } from './layerOperationHarness.testing';
import { traceVectorLayer } from './vectorTraceController';

describe('vector Trace/Fill pixels', () => {
  it('fills an intersecting open path without painting its tails and retains vector geometry', async () => {
    const layer = createVectorLayer({ id: 'v', name: 'Vector' });
    layer.fillEnabled = true;
    layer.strokeEnabled = false;
    layer.fillColor = '#ff0000';
    layer.paths = [
      {
        id: 'p',
        name: null,
        isClosed: false,
        points: anchorsToBezierPoints([
          { x: -20, y: -20 },
          { x: 20, y: 20 },
          { x: -20, y: 20 },
          { x: 20, y: -20 },
        ]),
      },
    ];
    const h = createLayerOperationHarness(documentFrom([layer], 'v'), { backend: createDomRasterBackend() });
    const doc = h.document();
    try {
      expect(
        await traceVectorLayer({
          document: doc,
          layerId: 'v',
          appearance: true,

          ctx: h.ctx,
          backend: h.backend,
          id: 'output',
          anchor: stackTopAnchor(h.ctx.projectId),
          isCurrent: () => h.document() === doc,
        })
      ).toEqual({ status: 'committed' });
      const entry = h.installed[0]!;
      const rgba = (x: number, y: number) =>
        Array.from(
          entry.surface.ctx.getImageData(Math.round(x - entry.rect.x), Math.round(y - entry.rect.y), 1, 1).data
        );
      expect(rgba(0, 10)).toEqual([255, 0, 0, 255]);
      expect(rgba(0, -10)[3]).toBe(0);
      expect(h.document().stacks.vector).toBe(doc.stacks.vector);
      expect(h.document().selectedLayerId).toBe('v');
    } finally {
      h.ctx.dispose();
      h.layers.dispose();
    }
  });
  it('renders independent fill and stroke colors and tapers an open stroke', async () => {
    const layer = createVectorLayer({ id: 'v', name: 'Vector' });
    layer.fillEnabled = true;
    layer.fillColor = '#00ff00';
    layer.strokeColor = '#0000ff';
    layer.strokeWidth = 8;
    layer.paths = [
      {
        id: 'p',
        name: null,
        isClosed: true,
        points: anchorsToBezierPoints([
          { x: 0, y: 0 },
          { x: 100, y: 0 },
          { x: 100, y: 100 },
          { x: 0, y: 100 },
        ]),
      },
    ];
    const h = createLayerOperationHarness(documentFrom([layer], 'v'), { backend: createDomRasterBackend() });
    const doc = h.document();
    try {
      expect(
        (
          await traceVectorLayer({
            document: doc,
            layerId: 'v',
            appearance: true,

            ctx: h.ctx,
            backend: h.backend,
            id: 'output',
            anchor: stackTopAnchor(h.ctx.projectId),
            isCurrent: () => h.document() === doc,
          })
        ).status
      ).toBe('committed');
      const entry = h.installed[0]!;
      const pixel = (x: number, y: number) =>
        Array.from(entry.surface.ctx.getImageData(x - entry.rect.x, y - entry.rect.y, 1, 1).data);
      expect(pixel(50, 50)).toEqual([0, 255, 0, 255]);
      expect(pixel(0, 50)).toEqual([0, 0, 255, 255]);
    } finally {
      h.ctx.dispose();
      h.layers.dispose();
    }
    layer.fillEnabled = false;
    layer.traceTaperEnds = true;
    layer.strokeWidth = 20;
    layer.paths = [
      {
        id: 'line',
        name: null,
        isClosed: false,
        points: anchorsToBezierPoints([
          { x: 0, y: 0 },
          { x: 100, y: 0 },
        ]),
      },
    ];
    const tapered = createLayerOperationHarness(documentFrom([layer], 'v'), { backend: createDomRasterBackend() });
    const before = tapered.document();
    try {
      expect(
        (
          await traceVectorLayer({
            document: before,
            layerId: 'v',

            ctx: tapered.ctx,
            backend: tapered.backend,
            id: 'taper',
            anchor: stackTopAnchor(tapered.ctx.projectId),
            isCurrent: () => tapered.document() === before,
          })
        ).status
      ).toBe('committed');
      const entry = tapered.installed[0]!;
      const alpha = (x: number) => entry.surface.ctx.getImageData(x - entry.rect.x, -entry.rect.y, 1, 1).data[3]!;
      expect(alpha(50)).toBe(255);
      expect(alpha(1)).toBeLessThan(alpha(50));
    } finally {
      tapered.ctx.dispose();
      tapered.layers.dispose();
    }
  });
});

it('uses each layer’s stroke color and width for individual path rendering', async () => {
  const paths = [
    {
      id: 'line',
      name: null,
      isClosed: false,
      points: anchorsToBezierPoints([
        { x: 0, y: 0 },
        { x: 100, y: 0 },
      ]),
    },
  ];
  const thin = { ...createVectorLayer({ id: 'thin', name: 'Thin' }), paths, strokeColor: '#ff0000', strokeWidth: 4 };
  const thick = {
    ...createVectorLayer({ id: 'thick', name: 'Thick' }),
    paths,
    strokeColor: '#0000ff',
    strokeWidth: 20,
  };
  for (const [id, expected, edgeAlpha] of [
    ['thin', [255, 0, 0, 255], 0],
    ['thick', [0, 0, 255, 255], 255],
  ] as const) {
    const h = createLayerOperationHarness(documentFrom([thin, thick], id), {
      backend: createDomRasterBackend(),
    });
    const document = h.document();
    try {
      expect(
        await traceVectorLayer({
          document,
          layerId: id,
          pathId: 'line',
          ctx: h.ctx,
          backend: h.backend,
          id: 'raster',
          anchor: stackTopAnchor(h.ctx.projectId),
          isCurrent: () => h.document() === document,
        })
      ).toEqual({ status: 'committed' });
      const entry = h.installed[0]!;
      const pixel = (y: number) =>
        Array.from(entry.surface.ctx.getImageData(50 - entry.rect.x, y - entry.rect.y, 1, 1).data);
      expect(pixel(0)).toEqual(expected);
      expect(pixel(4)[3]).toBe(edgeAlpha);
    } finally {
      h.ctx.dispose();
      h.layers.dispose();
    }
  }
});
