import { documentFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createVectorLayer } from '@workbench/canvas-engine/document/layerFactories';
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
          width: 4,
          color: '#000000',
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
            width: 8,
            color: '#0000ff',
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
            width: 20,
            color: '#000000',
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
