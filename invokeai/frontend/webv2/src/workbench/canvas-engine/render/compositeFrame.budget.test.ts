import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { FrameDamage } from '@workbench/canvas-engine/types';

import { PreviewStateController } from '@workbench/canvas-engine/controllers/previewStateController';
import { RasterController } from '@workbench/canvas-engine/controllers/rasterController';
import { createCanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import {
  createLargeTreeDocument,
  documentFrom,
  groupContract,
  layerContract,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { updateNodeValues } from '@workbench/canvas-engine/document/documentIndex';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { identity } from '@workbench/canvas-engine/math/mat2d';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { describe, expect, it, vi } from 'vitest';

import { createCompositeFrame } from './compositeFrame';
import * as compositor from './compositor';
import * as frameDemand from './frameDemand';

const SURFACE_BYTES = 10 * 10 * 4;

const adjusted = (id: string, x = 0): CanvasLayerContract =>
  layerContract(id, 'raster', {
    adjustments: [{ brightness: 0.5, contrast: 0, id: `${id}-bc`, isEnabled: true, type: 'brightness-contrast' }],
    source: { image: { height: 10, imageName: `${id}.png`, width: 10 }, type: 'image' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y: 0 },
  } as Partial<CanvasLayerContract>);

// Two base caches fit by default; their derived and group artifacts are steady overage.
const scene = (budgetBytes = SURFACE_BYTES * 2) => {
  const backend = createTestStubRasterBackend();
  const diagnostics = createCanvasDiagnostics(true);
  const raster = new RasterController({ backend, budgetBytes, diagnostics });
  const document = documentFrom([
    groupContract('group', [adjusted('member')], { opacity: 0.5 }),
    adjusted('plain', 40),
  ]);
  for (const id of ['member', 'plain']) {
    raster.layers.getOrCreateRect(id, { height: 10, width: 10, x: 0, y: 0 });
    raster.layers.publishPixels(id);
  }
  const viewport = createViewport();
  viewport.setViewportSize(100, 100, 1);
  const stores = createEngineStores();
  stores.checkerboard.set(false);
  const frame = createCompositeFrame({
    backend,
    derivedSurfaceCache: raster.derived,
    diagnostics,
    getAdjustedSurface: (layer, entry) => raster.getAdjustedSurface(layer, entry),
    getCheckerboardTile: () => backend.createSurface(1, 1),
    getGroupSurface: (scope, members, matrices, content) => raster.groups.get(scope, members, matrices, content),
    getMaskPatternTile: () => null,
    layerCache: raster.layers,
    previews: new PreviewStateController(),
    raster,
    rasterizeLayer: () => {
      throw new Error('caches are published');
    },
    stores,
    transformOverrides: new Map(),
    viewport,
  });
  const screen = backend.createSurface(100, 100);
  const draw = (damage?: FrameDamage) => frame.draw(screen, document, identity(), null, null, damage);
  const builds = () => {
    const snapshot = diagnostics.snapshot();
    return {
      derived: snapshot.derivedCacheMisses,
      groups: snapshot.groupSurfaceAllocations + snapshot.groupSurfaceRebuilds,
    };
  };
  return { builds, diagnostics, draw, raster };
};

describe('composite frame working set under steady overage', () => {
  it('keeps the visible artifacts across empty and bounded repaints instead of rebuilding them', () => {
    const { builds, draw, raster } = scene();
    draw();
    const afterFullFrame = builds();
    expect(afterFullFrame).toEqual({ derived: 2, groups: 1 });
    expect(raster.memory.snapshot().overageBytes).toBeGreaterThan(0);

    draw({ kind: 'regions', regions: [{ layerId: 'plain', rect: { height: 10, width: 10, x: 5000, y: 0 } }] });
    draw({ kind: 'regions', regions: [{ layerId: 'plain', rect: { height: 2, width: 2, x: 0, y: 0 } }] });
    expect(raster.memory.snapshot()).toMatchObject({ derivedBytes: SURFACE_BYTES * 2, groupBytes: SURFACE_BYTES });

    draw();
    expect(builds()).toEqual(afterFullFrame);
  });

  it('refreshes only the damaged region of a stable group surface while a member is painted', () => {
    const { diagnostics, draw, raster } = scene(SURFACE_BYTES * 100);
    draw();
    for (let tick = 0; tick < 5; tick += 1) {
      const rect = { height: 2, width: 2, x: tick, y: tick };
      raster.layers.publishPixels('member', rect);
      draw({ kind: 'regions', regions: [{ layerId: 'member', rect }] });
    }

    expect(diagnostics.snapshot()).toMatchObject({
      groupSurfaceAllocations: 1,
      groupSurfaceRebuilds: 0,
      groupSurfaceRefreshes: 5,
    });
  });
});

describe('frame description over 2,000 nodes', () => {
  it('prepares once per document and re-derives bounds only for the edited leaf', () => {
    const backend = createTestStubRasterBackend();
    const raster = new RasterController({ backend, diagnostics: createCanvasDiagnostics(true) });
    const viewport = createViewport();
    viewport.setViewportSize(100, 100, 1);
    const frame = createCompositeFrame({
      backend,
      derivedSurfaceCache: raster.derived,
      diagnostics: createCanvasDiagnostics(true),
      getAdjustedSurface: () => null,
      getCheckerboardTile: () => backend.createSurface(1, 1),
      getGroupSurface: () => null,
      getMaskPatternTile: () => null,
      layerCache: raster.layers,
      previews: new PreviewStateController(),
      raster,
      rasterizeLayer: () => undefined,
      stores: createEngineStores(),
      transformOverrides: new Map(),
      viewport,
    });
    const screen = backend.createSurface(100, 100);
    const composite = vi.spyOn(compositor, 'compositeDocument');
    const description = () => composite.mock.calls.at(-1)![4].preparation;
    const bounds = vi.spyOn(frameDemand, 'committedLeafBounds');
    const document = createLargeTreeDocument(2_000);

    frame.draw(screen, document, identity(), null, null);
    const leafCount = bounds.mock.calls.length;
    const first = description();
    expect(leafCount).toBeGreaterThan(1_000);
    frame.draw(screen, document, identity(), null, null, { kind: 'none' });
    expect(description()?.leaves).toBe(first?.leaves);
    expect(bounds).toHaveBeenCalledTimes(leafCount);

    const edited = {
      ...document,
      stacks: updateNodeValues(document.stacks, new Map([['l7', (node) => ({ ...node, opacity: 0.5 })]])),
    };
    frame.draw(screen, edited, identity(), null, null);
    expect(description()?.document).toBe(edited);
    expect(bounds.mock.calls.slice(leafCount).map(([leaf]) => leaf.id)).toEqual(['l7']);
    raster.dispose();
  });
});
