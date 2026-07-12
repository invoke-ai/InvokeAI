import type { CanvasCompositeExportGuard } from '@workbench/canvas-engine/engine';

import { describe, expect, it, vi } from 'vitest';

import { prepareSelectObjectSource, processSelectObjectSource } from './layerImageResult';

const rect = { height: 12, width: 16, x: -5, y: 7 };
const guard = {
  bbox: rect,
  candidates: [],
  documentFingerprint: 'document:1',
  documentGeneration: 1,
  participants: [],
  projectId: 'p1',
} satisfies CanvasCompositeExportGuard;

describe('Select Object image processing', () => {
  it('carries exact upload dimensions into the prepared source', async () => {
    await expect(
      prepareSelectObjectSource({
        exportComposite: () => Promise.resolve({ blob: new Blob(), guard, rect, status: 'ok' }),
        uploadIntermediate: () => Promise.resolve({ height: 12, imageName: 'source.png', width: 16 }),
      })
    ).resolves.toEqual({
      source: { guard, height: 12, imageName: 'source.png', rect, width: 16 },
      status: 'ready',
    });
  });

  it.each([
    { height: 12, width: 15 },
    { height: 12.5, width: 16 },
    { height: 0, width: 16 },
    { height: Number.NaN, width: 16 },
  ])('rejects upload dimensions that do not exactly match the bbox: $width x $height', async (dimensions) => {
    await expect(
      prepareSelectObjectSource({
        exportComposite: () => Promise.resolve({ blob: new Blob(), guard, rect, status: 'ok' }),
        uploadIntermediate: () => Promise.resolve({ ...dimensions, imageName: 'source.png' }),
      })
    ).resolves.toMatchObject({ status: 'dimension-mismatch' });
  });

  it('rejects a visual input made empty by export-local conversion before enqueue', async () => {
    const runGraph = vi.fn();
    await expect(
      processSelectObjectSource({
        applyPolygonRefinement: false,
        input: {
          bbox: null,
          excludePoints: [],
          includePoints: [{ x: 100, y: 100 }],
          type: 'visual',
        },
        invert: false,
        model: 'segment-anything-2-large',
        runGraph,
        source: { guard, height: 12, imageName: 'source.png', rect, width: 16 },
      })
    ).resolves.toEqual({ status: 'invalid-input' });
    expect(runGraph).not.toHaveBeenCalled();
  });

  it('rejects a prepared source whose threaded upload dimensions no longer match its bbox', async () => {
    const runGraph = vi.fn();
    await expect(
      processSelectObjectSource({
        applyPolygonRefinement: false,
        input: { prompt: 'cat', type: 'prompt' },
        invert: false,
        model: 'segment-anything-2-large',
        runGraph,
        source: { guard, height: 12, imageName: 'source.png', rect, width: 15 },
      })
    ).resolves.toMatchObject({ status: 'dimension-mismatch' });
    expect(runGraph).not.toHaveBeenCalled();
  });

  it.each([
    { height: 12, width: 15 },
    { height: 11.5, width: 16 },
    { height: Number.POSITIVE_INFINITY, width: 16 },
  ])('rejects mismatched SAM ImageOutput metadata without returning a stretchable preview', async (dimensions) => {
    await expect(
      processSelectObjectSource({
        applyPolygonRefinement: false,
        input: { prompt: 'cat', type: 'prompt' },
        invert: false,
        model: 'segment-anything-2-large',
        runGraph: () => Promise.resolve({ ...dimensions, imageName: 'mask.png', origin: 'test' }),
        source: { guard, height: 12, imageName: 'source.png', rect, width: 16 },
      })
    ).resolves.toMatchObject({ status: 'dimension-mismatch' });
  });
});
