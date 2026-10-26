import type { CanvasAdjustmentEntry } from '@workbench/canvas-engine/contracts';

import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { describe, expect, it } from 'vitest';

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
