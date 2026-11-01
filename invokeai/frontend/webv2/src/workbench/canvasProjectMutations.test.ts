import type { CanvasAdjustmentEntry, CanvasStagingCandidateContract } from '@workbench/canvas-engine/contracts';

import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { describe, expect, it } from 'vitest';

import { createEmptyCanvasDocument } from './canvasMigration';
import { applyCanvasProjectMutation } from './canvasProjectMutations';
import { createEmptyPaintLayer } from './widgets/layers/layerOps';
import { createDraftProject } from './workbenchState';

const noise: CanvasAdjustmentEntry = {
  amount: 0.3,
  distribution: 'uniform',
  grain: 'color',
  id: 'noise',
  isEnabled: true,
  response: 'midtones',
  seed: 9,
  size: 3,
  type: 'noise',
};

describe('canvas schema adoption', () => {
  it('moves a v4 canvas to schema 5 when a noise adjustment is added, and keeps 5 once it is removed', () => {
    const project = createDraftProject([]);
    const v4 = {
      ...project,
      canvas: { ...project.canvas, document: { ...project.canvas.document, version: 4 as const }, version: 4 as const },
    };

    const added = applyCanvasProjectMutation(v4, {
      anchor: stackTopAnchor(project.id),
      layer: createEmptyPaintLayer('Grain', 'grain'),
      type: 'addCanvasLayer',
    });
    expect(added.canvas.version).toBe(4);

    const grained = applyCanvasProjectMutation(added, {
      config: { adjustments: [noise], layerType: 'raster' },
      id: 'grain',
      type: 'updateCanvasLayerConfig',
    });
    expect(grained.canvas.version).toBe(5);
    expect(grained.canvas.document.version).toBe(5);

    const cleared = applyCanvasProjectMutation(grained, { ids: ['grain'], type: 'removeCanvasLayers' });
    expect(cleared.canvas.version).toBe(5);
    expect(cleared.canvas.document.version).toBe(5);
  });

  it.each<CanvasAdjustmentEntry>([
    { amount: 30, id: 'clarity', isEnabled: false, size: 40, type: 'clarity' },
    { amount: 30, id: 'texture', isEnabled: false, size: 8, type: 'texture' },
    { amount: 30, id: 'dehaze', isEnabled: false, type: 'dehaze' },
  ])('moves a v4 canvas to schema 5 when a disabled $type adjustment is added', (entry) => {
    const project = createDraftProject([]);
    const v4 = {
      ...project,
      canvas: { ...project.canvas, document: { ...project.canvas.document, version: 4 as const }, version: 4 as const },
    };
    const added = applyCanvasProjectMutation(v4, {
      anchor: stackTopAnchor(project.id),
      layer: createEmptyPaintLayer('Detail', 'detail'),
      type: 'addCanvasLayer',
    });

    const adjusted = applyCanvasProjectMutation(added, {
      config: { adjustments: [entry], layerType: 'raster' },
      id: 'detail',
      type: 'updateCanvasLayerConfig',
    });
    expect(adjusted.canvas.version).toBe(5);
    expect(adjusted.canvas.document.version).toBe(5);
  });
});

describe('restore points across document sessions', () => {
  const result: CanvasStagingCandidateContract = {
    height: 64,
    imageName: 'result.png',
    imageUrl: '/api/v1/images/i/result.png/full',
    placement: { height: 64, opacity: 1, width: 64, x: 8, y: 8 },
    queuedAt: '2026-10-30T00:00:00.000Z',
    sourceQueueItemId: 'queue-1',
    thumbnailUrl: '/api/v1/images/i/result.png/thumbnail',
    width: 64,
  };

  /** A canvas with a Clear canvas point, then a Discard all point of `result`, its staging discarded since. */
  const withBothPoints = () => {
    const draft = createDraftProject([]);
    let project = {
      ...draft,
      canvas: {
        ...draft.canvas,
        stagingArea: { ...draft.canvas.stagingArea, pendingImageIds: [result.imageName], pendingImages: [result] },
      },
    };
    for (const kind of ['clear-canvas', 'discard-results'] as const) {
      project = applyCanvasProjectMutation(project, {
        createdAt: 'now',
        id: kind,
        includeStaging: true,
        kind,
        name: kind,
        type: 'saveCanvasSnapshot',
      });
    }
    return applyCanvasProjectMutation(project, { type: 'discardAllStagedImages' });
  };
  const pointIds = (project: ReturnType<typeof createDraftProject>) => project.canvas.snapshots.map(({ id }) => id);

  it('drops Discard all points with the document they belong to, keeping Clear canvas points', () => {
    const replaced = applyCanvasProjectMutation(withBothPoints(), {
      document: createEmptyCanvasDocument(),
      type: 'replaceCanvasDocument',
    });
    expect(pointIds(replaced)).toEqual(['clear-canvas']);

    const moved = applyCanvasProjectMutation(withBothPoints(), {
      height: 1024,
      offsetX: 16,
      offsetY: 0,
      type: 'resizeCanvasDocument',
      width: 1024,
    });
    expect(moved.canvas.document.width).toBe(1024);
    expect(pointIds(moved)).toEqual(['clear-canvas']);
    const grown = applyCanvasProjectMutation(withBothPoints(), {
      height: 1024,
      type: 'resizeCanvasDocument',
      width: 1024,
    });
    expect(grown.canvas.document.width).toBe(1024);
    expect(pointIds(grown)).toEqual(['clear-canvas', 'discard-results']);

    const kept = applyCanvasProjectMutation(withBothPoints(), {
      keepCurrentAs: { createdAt: 'now', id: 'undo', name: 'Before restoring' },
      snapshotId: 'clear-canvas',
      type: 'restoreCanvasSnapshot',
    });
    expect(pointIds(kept)).toEqual(['undo']);

    const restored = applyCanvasProjectMutation(withBothPoints(), {
      snapshotId: 'clear-canvas',
      type: 'restoreCanvasSnapshot',
    });
    expect(pointIds(restored)).toEqual(['clear-canvas']);
    // The results come back only with the document they were placed on.
    expect(restored.canvas.stagingArea.pendingImages).toEqual([result]);
  });
});
