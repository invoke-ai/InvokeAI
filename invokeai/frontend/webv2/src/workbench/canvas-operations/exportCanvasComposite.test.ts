import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { RasterCompositeExportResult } from '@workbench/canvas-engine/exportRasterComposite';
import type { Rect } from '@workbench/canvas-engine/types';

import { describe, expect, it, vi } from 'vitest';

import { exportCanvasComposite } from './exportCanvasComposite';

const defaultRect: Rect = { height: 456, width: 123, x: -4, y: 8 };
const defaultBlob = new Blob(['pixels'], { type: 'image/png' });

const createDocument = (bbox: Rect): CanvasDocumentContractV2 => ({ bbox }) as unknown as CanvasDocumentContractV2;

const createHarness = (
  options: {
    document?: CanvasDocumentContractV2 | null;
    exportResult?: RasterCompositeExportResult;
    onFlush?: () => void;
  } = {}
) => {
  const order: string[] = [];
  let document = options.document === undefined ? createDocument(defaultRect) : options.document;
  const exportResult = options.exportResult ?? { blob: defaultBlob, rect: defaultRect, status: 'ok' };
  const flushPendingUploads = vi.fn(() => {
    order.push('flush');
    options.onFlush?.();
    return Promise.resolve();
  });
  const getDocument = vi.fn(() => {
    order.push('document');
    return document;
  });
  const exportRasterComposite = vi.fn<CanvasEngine['exports']['exportRasterComposite']>(() => {
    order.push('export');
    return Promise.resolve(exportResult);
  });
  const engine = {
    document: { getDocument },
    exports: { exportRasterComposite },
    lifecycle: { flushPendingUploads },
  } as unknown as CanvasEngine;

  return {
    engine,
    exportRasterComposite,
    order,
    setDocument: (next: CanvasDocumentContractV2 | null) => {
      document = next;
    },
  };
};

describe('exportCanvasComposite', () => {
  it('flushes pending uploads before reading the document and exporting', async () => {
    const harness = createHarness();

    await expect(exportCanvasComposite(harness.engine, 'canvas')).resolves.toEqual({
      blob: defaultBlob,
      rect: defaultRect,
      status: 'ok',
    });

    expect(harness.order).toEqual(['flush', 'document', 'export']);
    expect(harness.exportRasterComposite).toHaveBeenCalledWith({ bounds: 'content' });
  });

  it('exports the post-flush document bbox for the bbox region', async () => {
    const preFlushRect: Rect = { height: 20, width: 10, x: 0, y: 0 };
    const postFlushRect: Rect = { height: 40, width: 30, x: 5, y: 6 };
    let harness: ReturnType<typeof createHarness>;
    harness = createHarness({
      document: createDocument(preFlushRect),
      onFlush: () => harness.setDocument(createDocument(postFlushRect)),
    });

    await exportCanvasComposite(harness.engine, 'bbox');

    expect(harness.exportRasterComposite).toHaveBeenCalledWith({ bounds: 'rect', rect: postFlushRect });
  });

  it('returns not-ready without exporting when there is no post-flush document', async () => {
    const harness = createHarness({ document: null });

    await expect(exportCanvasComposite(harness.engine, 'bbox')).resolves.toEqual({ status: 'not-ready' });

    expect(harness.exportRasterComposite).not.toHaveBeenCalled();
  });

  it.each(['empty', 'stale', 'not-ready', 'over-budget'] as const)('passes %s through verbatim', async (status) => {
    const harness = createHarness({ exportResult: { status } });

    await expect(exportCanvasComposite(harness.engine, 'canvas')).resolves.toEqual({ status });
  });
});
