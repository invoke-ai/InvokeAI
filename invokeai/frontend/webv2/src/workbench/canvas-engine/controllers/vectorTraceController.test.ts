import { documentFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createVectorLayer } from '@workbench/canvas-engine/document/layerFactories';
import { anchorsToBezierPoints, approximateBezierPath } from '@workbench/canvas-engine/math/bezierPath';
import { getTraceLineBounds } from '@workbench/canvas-engine/math/vectorTrace';
import { describe, expect, it } from 'vitest';

import { createLayerOperationHarness } from './layerOperationHarness.testing';
import { traceVectorLayer } from './vectorTraceController';

const setup = (historyBytes?: number) => {
  const layer = createVectorLayer({ id: 'v', name: 'Vector' });
  layer.paths = [
    {
      id: 'p',
      name: null,
      isClosed: true,
      points: anchorsToBezierPoints([
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
      ]),
    },
  ];
  layer.fillEnabled = true;
  layer.strokeEnabled = true;
  layer.fillColor = '#ff0000';
  const h = createLayerOperationHarness(documentFrom([layer], 'v'), { historyBytes });
  const doc = h.document();
  const createPath = () =>
    ({ moveTo() {}, lineTo() {}, closePath() {}, quadraticCurveTo() {}, bezierCurveTo() {} }) as unknown as Path2D;
  const run = (signal?: AbortSignal, isCurrent = () => h.document() === doc) =>
    traceVectorLayer({
      document: doc,
      layerId: 'v',
      appearance: true,
      width: 4,
      color: '#000000',
      id: 'output',
      ctx: h.ctx,
      backend: h.backend,
      anchor: stackTopAnchor(h.ctx.projectId),
      signal,
      isCurrent,
      createPath,
    });
  return { h, doc, run };
};
describe('vector trace transaction', () => {
  it('publishes one raster while preserving selection, supports Undo/Redo and releases reservations', async () => {
    const { h, doc, run } = setup();
    expect(await run()).toEqual({ status: 'committed' });
    expect(h.document().selectedLayerId).toBe('v');
    expect(h.document().stacks.raster[0]?.id).toBe('output');
    expect(h.document().stacks.vector).toBe(doc.stacks.vector);
    expect(h.reservedBytes()).toBe(0);
    await h.history.undo();
    expect(h.document().stacks.raster).toHaveLength(0);
    await h.history.redo();
    expect(h.document().stacks.raster[0]?.id).toBe('output');
    expect(h.document().selectedLayerId).toBe('v');
  });
  it('does not publish or retain allocations after cancel, stale input or budget refusal', async () => {
    const a = setup(),
      controller = new AbortController(),
      pending = a.run(controller.signal);
    controller.abort();
    expect(await pending).toEqual({ status: 'aborted' });
    expect(a.h.installed).toHaveLength(0);
    expect(a.h.reservedBytes()).toBe(0);
    const b = setup();
    let current = true;
    const stale = b.run(undefined, () => current);
    current = false;
    expect(await stale).toEqual({ status: 'stale' });
    expect(b.h.installed).toHaveLength(0);
    const c = setup(1);
    expect(await c.run()).toEqual({ status: 'over-budget' });
    expect(c.h.reservedBytes()).toBe(0);
  });
  it('bounds tension control points rather than only the original samples', () => {
    const coordinates = approximateBezierPath(
      anchorsToBezierPoints([
        { x: 0, y: 0 },
        { x: 12000, y: 0 },
        { x: 12000, y: 12000 },
      ]),
      false
    );
    const bounds = getTraceLineBounds(coordinates);
    expect(bounds.x + bounds.width).toBeGreaterThan(12004.05);
  });
});
