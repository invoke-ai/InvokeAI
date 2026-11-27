import type { CanvasAdjustmentsContract, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { GroupSurfaceResult } from '@workbench/canvas-engine/render/groupSurfaceCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { createCanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { identity } from '@workbench/canvas-engine/math/mat2d';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { applyAdjustments, toSpatialPlan } from '@workbench/canvas-engine/render/adjustments';
import { compositeDocument } from '@workbench/canvas-engine/render/compositor';
import { createDerivedSurfaceCache } from '@workbench/canvas-engine/render/derivedSurfaceCache';
import { createGroupSurfaceCache } from '@workbench/canvas-engine/render/groupSurfaceCache';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { planBaseRasterComposite, renderRasterComposite } from '@workbench/canvas-engine/render/rasterComposite';
import { evaluateSpatialSteps } from '@workbench/canvas-engine/render/spatial/evaluateSpatialSteps';
import {
  createSpatialAdjustmentService,
  type SpatialAdjustmentService,
} from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { afterEach, describe, expect, it, vi } from 'vitest';

const WIDTH = 64;
const HEIGHT = 64;
const BBOX: Rect = { height: HEIGHT, width: WIDTH, x: 0, y: 0 };

const raster = (id: string, overrides: Partial<CanvasRasterLayerContractV2> = {}): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: { height: HEIGHT, imageName: id, width: WIDTH }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
  ...overrides,
});

const gammaStack = (id: string): CanvasAdjustmentsContract => [
  { gamma: 2, id, inBlack: 0, inWhite: 255, isEnabled: true, outBlack: 0, outWhite: 255, type: 'levels' },
];
const invertStack = (id: string): CanvasAdjustmentsContract => [{ id, isEnabled: true, type: 'invert' }];
const noiseStack = (id: string): CanvasAdjustmentsContract => [
  {
    amount: 0.2,
    distribution: 'gaussian',
    id,
    isEnabled: true,
    grain: 'color',
    response: 'flat',
    size: 1,
    seed: 4242,
    type: 'noise',
  },
];
const RAW_MEMBERS = () => ({ state: 'exact' as const, surface: null });

const worstDifference = (a: Uint8ClampedArray, b: Uint8ClampedArray): number => {
  let worst = 0;
  for (let i = 0; i < a.length; i += 1) {
    worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
  }
  return worst;
};

const sceneFor = (fills: Record<string, string>) => {
  const backend = createDomRasterBackend();
  const caches = createLayerCacheStore(backend);
  for (const [id, color] of Object.entries(fills)) {
    caches.growToRect(id, BBOX);
    const surface = caches.get(id)!.surface;
    surface.ctx.setTransform(1, 0, 0, 1, 0, 0);
    surface.ctx.fillStyle = color;
    surface.ctx.fillRect(0, 0, WIDTH, HEIGHT);
    caches.publishPixels(id);
  }
  return {
    backend,
    caches,
    getLayerSurface: (layerId: string) => {
      const entry = caches.get(layerId)!;
      return Promise.resolve({ rect: entry.rect, release: () => undefined, surface: entry.surface });
    },
  };
};

const centerPixel = (surface: RasterSurface): number[] => [
  ...surface.ctx.getImageData(WIDTH / 2, HEIGHT / 2, 1, 1).data,
];

const docWith = (nodes: Parameters<typeof stacksFrom>[0]) => ({
  background: 'transparent' as const,
  bbox: BBOX,
  height: HEIGHT,
  selectedLayerId: null,
  stacks: stacksFrom(nodes),
  version: 3 as const,
  width: WIDTH,
});

describe('group adjustment composite', () => {
  it('applies the stack to the group composite, not to each member (the per-leaf bake is measurably wrong)', async () => {
    // 50% red over white inside the group → pink (255,~127,~127);
    // gamma-2 on the COMPOSITE lifts g/b to ~180. Per-member baking would leave g/b near ~127 (gamma fixes 0 and 255),
    // 50+ levels away.
    const document = docWith([
      groupContract('g', [raster('red', { opacity: 0.5 }), raster('white')], {
        adjustments: gammaStack('ga'),
      }),
    ]);
    const scene = sceneFor({ red: '#ff0000', white: '#ffffff' });
    const entry = planBaseRasterComposite(document, BBOX);
    expect(entry.groupScopes).toHaveLength(1);

    const surface = await renderRasterComposite(entry, scene);
    const [r, g, b, a] = centerPixel(surface);
    expect(a).toBe(255);
    expect(r).toBe(255);
    expect(Math.abs(g! - 180)).toBeLessThanOrEqual(3);
    expect(Math.abs(b! - 180)).toBeLessThanOrEqual(3);
  });

  it('draws the same pixels on screen (group surface cache) as the export renderer', async () => {
    const document = docWith([
      groupContract('g', [raster('red', { blendMode: 'multiply', opacity: 0.6 }), raster('white')], {
        adjustments: gammaStack('ga'),
      }),
      raster('under'),
    ]);
    const scene = sceneFor({ red: '#ff4040', under: '#2040c0', white: '#c0c0c0' });
    const groupSurfaces = createGroupSurfaceCache({
      createSurface: (w, h) => scene.backend.createSurface(w, h),
      getAdjustedSurface: RAW_MEMBERS,
      damageSince: (id, version) => scene.caches.damageSince(id, version),
      getCacheEntry: (id) => scene.caches.get(id),
    });

    const screen = scene.backend.createSurface(WIDTH, HEIGHT);
    compositeDocument(screen, document, scene.caches, identity(), {
      backend: scene.backend,
      groupSurface: (scope, members, matrices, content) => groupSurfaces.get(scope, members, matrices, content),
    });
    const exported = await renderRasterComposite(planBaseRasterComposite(document, BBOX), scene);

    const screenPx = centerPixel(screen);
    const exportPx = centerPixel(exported);
    for (let channel = 0; channel < 4; channel += 1) {
      expect(Math.abs(screenPx[channel]! - exportPx[channel]!)).toBeLessThanOrEqual(1);
    }
    const again = scene.backend.createSurface(WIDTH, HEIGHT);
    compositeDocument(again, document, scene.caches, identity(), {
      backend: scene.backend,
      groupSurface: (scope, members, matrices, content) => groupSurfaces.get(scope, members, matrices, content),
    });
    expect(centerPixel(again)).toEqual(screenPx);
  });

  it('applies group opacity to the isolated composite, matching the export renderer', async () => {
    // Opaque red over opaque white inside a 50%-opacity group, over a blue base: the GROUP composite is pure red (white
    // hidden underneath), so the result is red 50% over blue. Per-leaf opacity instead would let white bleed through
    // inside the group.
    const document = docWith([groupContract('g', [raster('red'), raster('white')], { opacity: 0.5 }), raster('under')]);
    const scene = sceneFor({ red: '#ff0000', under: '#0000ff', white: '#ffffff' });
    const plan = planBaseRasterComposite(document, BBOX);
    expect(plan.groupScopes).toHaveLength(1);
    const exported = await renderRasterComposite(plan, scene);
    const [r, g, b] = centerPixel(exported);
    expect(Math.abs(r! - 128)).toBeLessThanOrEqual(2);
    expect(g).toBe(0);
    expect(Math.abs(b! - 128)).toBeLessThanOrEqual(2);

    const groupSurfaces = createGroupSurfaceCache({
      createSurface: (w, h) => scene.backend.createSurface(w, h),
      getAdjustedSurface: RAW_MEMBERS,
      damageSince: (id, version) => scene.caches.damageSince(id, version),
      getCacheEntry: (id) => scene.caches.get(id),
    });
    const screen = scene.backend.createSurface(WIDTH, HEIGHT);
    compositeDocument(screen, document, scene.caches, identity(), {
      backend: scene.backend,
      groupSurface: (scope, members, matrices, content) => groupSurfaces.get(scope, members, matrices, content),
    });
    const screenPx = centerPixel(screen);
    const exportPx = centerPixel(exported);
    for (let channel = 0; channel < 4; channel += 1) {
      expect(Math.abs(screenPx[channel]! - exportPx[channel]!)).toBeLessThanOrEqual(1);
    }
  });

  it('keeps identity and pass-through groups on the flat path and keys the plan by group stacks', () => {
    const flat = docWith([groupContract('g', [raster('white')])]);
    expect(planBaseRasterComposite(flat, BBOX).groupScopes).toBeUndefined();

    const adjusted = docWith([groupContract('g', [raster('white')], { adjustments: gammaStack('ga') })]);
    const inverted = docWith([groupContract('g', [raster('white')], { adjustments: invertStack('ia') })]);
    expect(planBaseRasterComposite(adjusted, BBOX).key).not.toBe(planBaseRasterComposite(flat, BBOX).key);
    expect(planBaseRasterComposite(adjusted, BBOX).key).not.toBe(planBaseRasterComposite(inverted, BBOX).key);
  });
});

describe('group-scoped previews and partial refresh', () => {
  interface GroupedSceneOptions {
    memberRect?: Rect;
    member?: Partial<CanvasRasterLayerContractV2>;
    group?: Parameters<typeof groupContract>[2];
    nested?: boolean;
  }

  const groupedScene = ({
    group = { opacity: 0.5 },
    member = {},
    memberRect = BBOX,
    nested = false,
  }: GroupedSceneOptions = {}) => {
    const scene = sceneFor({ under: '#0000ff' });
    scene.caches.growToRect('member', memberRect);
    const memberSurface = scene.caches.get('member')!.surface;
    memberSurface.ctx.fillStyle = '#00ff00';
    memberSurface.ctx.fillRect(0, 0, memberRect.width, memberRect.height);
    scene.caches.publishPixels('member');
    const diagnostics = createCanvasDiagnostics(true);
    const groupSurfaces = createGroupSurfaceCache({
      createSurface: (w, h) => scene.backend.createSurface(w, h),
      damageSince: (id, version) => scene.caches.damageSince(id, version),
      diagnostics,
      getAdjustedSurface: RAW_MEMBERS,
      getCacheEntry: (id) => scene.caches.get(id),
    });
    const memberNode = raster('member', member);
    const document = docWith([
      groupContract('g', [nested ? groupContract('inner', [memberNode], { opacity: 0.8 }) : memberNode], group),
      raster('under'),
    ]);
    const composite = (options: Partial<Parameters<typeof compositeDocument>[4]> = {}) => {
      const screen = scene.backend.createSurface(WIDTH, HEIGHT);
      compositeDocument(screen, document, scene.caches, identity(), {
        backend: scene.backend,
        groupSurface: (scope, members, matrices, content) => groupSurfaces.get(scope, members, matrices, content),
        ...options,
      });
      return screen;
    };
    const paintDot = (bounded: boolean) => {
      const entry = scene.caches.get('member')!;
      const dot = { height: 1, width: 1, x: 3, y: 4 };
      entry.surface.ctx.fillStyle = '#ff00ff';
      entry.surface.ctx.fillRect(dot.x, dot.y, dot.width, dot.height);
      scene.caches.publishPixels('member', bounded ? dot : undefined);
    };
    return { composite, diagnostics, groupSurfaces, paintDot, scene };
  };

  const allPixels = (surface: RasterSurface): number[] => [...surface.ctx.getImageData(0, 0, WIDTH, HEIGHT).data];

  it('draws a member filter preview inside its group, under the group opacity', () => {
    const { composite, scene } = groupedScene();
    const committed = centerPixel(composite());
    const previewSurface = scene.backend.createSurface(WIDTH, HEIGHT);
    previewSurface.ctx.fillStyle = '#ff0000';
    previewSurface.ctx.fillRect(0, 0, WIDTH, HEIGHT);

    const previewed = centerPixel(
      composite({ layerPreviews: new Map([['member', { rect: BBOX, surface: previewSurface }]]) })
    );

    expect(committed[1]).toBeGreaterThan(120);
    expect(committed[1]).toBeLessThan(135);
    expect(previewed[0]).toBeGreaterThan(120);
    expect(previewed[0]).toBeLessThan(135);
    expect(previewed[2]).toBeGreaterThan(120);
    expect(previewed[3]).toBe(255);
  });

  it.each<[string, GroupedSceneOptions]>([
    ['an untransformed member', {}],
    [
      'a member scaled 8x',
      {
        member: { transform: { rotation: 0, scaleX: 8, scaleY: 8, x: 0, y: 0 } },
        memberRect: { height: 8, width: 8, x: 0, y: 0 },
      },
    ],
    ['an inverted group', { group: { adjustments: invertStack('ga'), opacity: 0.5 } }],
    ['a group with levels', { group: { adjustments: gammaStack('ga') } }],
    ['a group with noise', { group: { adjustments: noiseStack('gn') } }],
    ['a member inside a nested group', { nested: true }],
  ])('refreshes a damaged region of %s to the same pixels as a full rebuild', (_, options) => {
    const refreshed = groupedScene(options);
    refreshed.composite();
    refreshed.paintDot(true);
    const pixels = allPixels(refreshed.composite());
    expect(refreshed.diagnostics.snapshot().groupSurfaceRefreshes).toBeGreaterThan(0);

    const rebuilt = groupedScene(options);
    rebuilt.paintDot(false);
    expect(pixels).toEqual(allPixels(rebuilt.composite()));
  });
});

describe('layer adjustments in export', () => {
  const adjustedScene = () => {
    const scene = sceneFor({});
    scene.caches.growToRect('graded', { height: 16, width: 16, x: 0, y: 0 });
    const ctx = scene.caches.get('graded')!.surface.ctx;
    for (let x = 0; x < 16; x += 1) {
      ctx.fillStyle = `rgb(${x * 16}, ${255 - x * 16}, 128)`;
      ctx.fillRect(x, 0, 1, 16);
    }
    scene.caches.publishPixels('graded');
    const layer = raster('graded', {
      adjustments: gammaStack('ga'),
      transform: { rotation: 0, scaleX: 3, scaleY: 3, x: 5, y: 7 },
    });
    return { document: docWith([layer]), layer, scene };
  };

  it('matches the display pixel for pixel by adjusting layer-local pixels before resampling', async () => {
    const { document, layer, scene } = adjustedScene();
    const adjusted = createAdjustedSurfaceCache(scene.backend);
    const screen = scene.backend.createSurface(WIDTH, HEIGHT);
    compositeDocument(screen, document, scene.caches, identity(), {
      adjustedSurface: (_layer, entry) => adjusted.get(layer.id, entry, layer.adjustments),
      backend: scene.backend,
    });
    const exported = await renderRasterComposite(planBaseRasterComposite(document, BBOX), scene);

    const display = screen.ctx.getImageData(0, 0, WIDTH, HEIGHT).data;
    const exportedPixels = exported.ctx.getImageData(0, 0, WIDTH, HEIGHT).data;
    let worst = 0;
    for (let i = 0; i < display.length; i += 1) {
      worst = Math.max(worst, Math.abs(display[i]! - exportedPixels[i]!));
    }
    expect(worst).toBeLessThanOrEqual(1);
  });

  it("draws the display's adjusted copy instead of adjusting again", async () => {
    const { document, layer, scene } = adjustedScene();
    const adjusted = createAdjustedSurfaceCache(scene.backend);
    const display = adjusted.get(layer.id, scene.caches.get(layer.id)!, layer.adjustments)!;
    const sizes: [number, number][] = [];
    const backend = {
      ...scene.backend,
      createSurface: (width: number, height: number) => {
        sizes.push([width, height]);
        return scene.backend.createSurface(width, height);
      },
    };
    const shared: RasterSurface[] = [];
    const exported = await renderRasterComposite(planBaseRasterComposite(document, BBOX), {
      ...scene,
      adjustedSurface: (_layerId, surface) => {
        shared.push(surface);
        return display;
      },
      backend,
    });

    expect(shared).toEqual([scene.caches.get(layer.id)!.surface]);
    expect(sizes).toEqual([[WIDTH, HEIGHT]]);
    const screen = scene.backend.createSurface(WIDTH, HEIGHT);
    compositeDocument(screen, document, scene.caches, identity(), {
      adjustedSurface: () => display,
      backend: scene.backend,
    });
    expect([...exported.ctx.getImageData(0, 0, WIDTH, HEIGHT).data]).toEqual([
      ...screen.ctx.getImageData(0, 0, WIDTH, HEIGHT).data,
    ]);
  });
});

describe('spatial group display', () => {
  const MEMBER: Rect = { height: 30, width: 40, x: 0, y: 0 };
  const AT = { x: 9, y: 7 };
  const translated = { rotation: 0, scaleX: 1, scaleY: 1, ...AT };
  const services: SpatialAdjustmentService[] = [];
  afterEach(() => {
    for (const service of services.splice(0)) {
      service.dispose();
    }
  });

  const spatialScene = () => {
    const scene = sceneFor({});
    scene.caches.growToRect('m', MEMBER);
    const ctx = scene.caches.get('m')!.surface.ctx;
    const gradient = ctx.createLinearGradient(0, 0, MEMBER.width, MEMBER.height);
    gradient.addColorStop(0, '#1d4ed8');
    gradient.addColorStop(0.5, '#e2e8f0');
    gradient.addColorStop(1, '#f59e0b');
    ctx.fillStyle = gradient;
    ctx.fillRect(3, 2, MEMBER.width - 7, MEMBER.height - 5);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(14, 9, 10, 8);
    scene.caches.publishPixels('m');
    const budget = new RasterMemoryBudgetController({ budgetBytes: 64 * 1024 * 1024 });
    const service = createSpatialAdjustmentService({
      reserve: (bytes) => budget.reserveOperation(bytes, { purpose: 'spatial-adjustment' }),
    });
    services.push(service);
    return { scene, service };
  };

  const pixelsOf = (surface: RasterSurface): Uint8ClampedArray =>
    surface.ctx.getImageData(0, 0, surface.width, surface.height).data;

  const roundTrip = (
    backend: ReturnType<typeof createDomRasterBackend>,
    pixels: Uint8ClampedArray,
    width: number,
    height: number
  ) => {
    const scratch = backend.createSurface(width, height);
    const image = scratch.ctx.createImageData(width, height);
    image.data.set(pixels);
    scratch.ctx.putImageData(image, 0, 0);
    return scratch;
  };

  const evaluateStack = (
    pixels: Uint8ClampedArray,
    width: number,
    height: number,
    stack: CanvasAdjustmentsContract,
    origin = { x: 0, y: 0 }
  ): Uint8ClampedArray => {
    const out = pixels.slice();
    for (const run of toSpatialPlan(stack)) {
      if (run.kind === 'per-pixel') {
        applyAdjustments({ data: out, height, width }, run.adjustments, origin);
      } else {
        expect(evaluateSpatialSteps(out, width, height, run.steps)).toEqual({ status: 'ok' });
      }
    }
    return out;
  };

  const memberOnDocument = (
    scene: ReturnType<typeof sceneFor>,
    member: RasterSurface = scene.caches.get('m')!.surface
  ) => {
    const document = scene.backend.createSurface(WIDTH, HEIGHT);
    document.ctx.drawImage(member.canvas, AT.x, AT.y);
    return document;
  };

  const settle = async (composite: () => void, done: () => boolean) => {
    await vi.waitFor(
      () => {
        composite();
        expect(done()).toBe(true);
      },
      { interval: 20, timeout: 10_000 }
    );
  };

  it('settles to the stack evaluated on the raw composite, independent of the buffer holding it', async () => {
    const stack: CanvasAdjustmentsContract = [
      { brightness: 0.1, contrast: 0.05, id: 'bc', isEnabled: true, type: 'brightness-contrast' },
      { amount: 70, id: 'clarity', isEnabled: true, size: 12, type: 'clarity' },
      ...noiseStack('grain'),
    ];
    const { scene, service } = spatialScene();
    const document = docWith([groupContract('g', [raster('m', { transform: translated })], { adjustments: stack })]);
    const results = new Map<string, GroupSurfaceResult | null>();
    const groupSurfaces = createGroupSurfaceCache({
      createSurface: (w, h) => scene.backend.createSurface(w, h),
      damageSince: (id, version) => scene.caches.damageSince(id, version),
      getAdjustedSurface: RAW_MEMBERS,
      getCacheEntry: (id) => scene.caches.get(id),
      spatial: { run: (request) => service.run(request) },
    });
    const screen = scene.backend.createSurface(WIDTH, HEIGHT);
    const composite = () => {
      screen.ctx.clearRect(0, 0, WIDTH, HEIGHT);
      compositeDocument(screen, document, scene.caches, identity(), {
        backend: scene.backend,
        groupSurface: (scope, members, matrices, content) => {
          const result = groupSurfaces.get(scope, members, matrices, content);
          results.set(scope.id, result);
          return result;
        },
      });
    };

    composite();
    expect(results.get('g')).toMatchObject({ rect: { height: 30, width: 40, ...AT }, state: 'pending' });
    await settle(composite, () => results.get('g')?.state === 'exact');

    const raw = pixelsOf(memberOnDocument(scene));
    const expected = pixelsOf(roundTrip(scene.backend, evaluateStack(raw, WIDTH, HEIGHT, stack), WIDTH, HEIGHT));
    expect(worstDifference(pixelsOf(screen), expected)).toBe(0);
    const perPixel = evaluateStack(
      raw,
      WIDTH,
      HEIGHT,
      stack.filter((entry) => entry.type !== 'clarity')
    );
    expect(worstDifference(expected, pixelsOf(roundTrip(scene.backend, perPixel, WIDTH, HEIGHT)))).toBeGreaterThan(0);
  });

  it("repaints a group when a member's spatial result lands", async () => {
    const memberStack: CanvasAdjustmentsContract = [
      { amount: 80, id: 'clarity', isEnabled: true, size: 12, type: 'clarity' },
    ];
    const member = raster('m', { adjustments: memberStack, transform: translated });
    const document = docWith([groupContract('g', [member], { adjustments: invertStack('inv') })]);
    const { scene, service } = spatialScene();
    let landed = false;
    const adjusted = createAdjustedSurfaceCache(scene.backend, createDerivedSurfaceCache(), {
      damageSince: (id, version) => scene.caches.damageSince(id, version),
      spatial: {
        onSettle: () => {
          landed = true;
          groupSurfaces.invalidate(['g']);
        },
        run: (request) => service.run(request),
      },
    });
    const groupSurfaces = createGroupSurfaceCache({
      createSurface: (w, h) => scene.backend.createSurface(w, h),
      damageSince: (id, version) => scene.caches.damageSince(id, version),
      getAdjustedSurface: (layer, entry) => {
        const surface = adjusted.get(layer.id, entry, layer.adjustments);
        return { state: surface ? adjusted.state(layer.id, entry, layer.adjustments) : 'exact', surface };
      },
      getCacheEntry: (id) => scene.caches.get(id),
    });
    const screen = scene.backend.createSurface(WIDTH, HEIGHT);
    let state: GroupSurfaceResult['state'] | undefined;
    const composite = () => {
      screen.ctx.clearRect(0, 0, WIDTH, HEIGHT);
      compositeDocument(screen, document, scene.caches, identity(), {
        backend: scene.backend,
        groupSurface: (scope, members, matrices, content) => {
          const result = groupSurfaces.get(scope, members, matrices, content);
          state = result?.state;
          return result;
        },
      });
    };
    const inverted = (pixels: Uint8ClampedArray) => {
      const out = pixels.slice();
      applyAdjustments({ data: out, height: HEIGHT, width: WIDTH }, invertStack('inv'), { x: 0, y: 0 });
      return pixelsOf(roundTrip(scene.backend, out, WIDTH, HEIGHT));
    };

    composite();
    expect(state).toBe('pending');
    const interim = pixelsOf(screen);
    expect(worstDifference(interim, inverted(pixelsOf(memberOnDocument(scene))))).toBe(0);
    await vi.waitFor(() => expect(landed).toBe(true), { timeout: 10_000 });
    composite();

    expect(state).toBe('exact');
    const source = scene.caches.get('m')!.surface;
    const memberResult = roundTrip(
      scene.backend,
      evaluateStack(pixelsOf(source), source.width, source.height, memberStack),
      source.width,
      source.height
    );
    const expected = inverted(pixelsOf(memberOnDocument(scene, memberResult)));
    expect(worstDifference(pixelsOf(screen), expected)).toBe(0);
    expect(worstDifference(interim, expected)).toBeGreaterThan(0);
  });

  it('refreshes an edited and moved member in place to the pixels a fresh build settles to', async () => {
    const { scene, service } = spatialScene();
    scene.caches.growToRect('b', { height: 8, width: 8, x: 0, y: 0 });
    const ctx = scene.caches.get('b')!.surface.ctx;
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(1, 1, 6, 6);
    scene.caches.publishPixels('b');
    const stack: CanvasAdjustmentsContract = [
      { amount: 70, id: 'clarity', isEnabled: true, size: 12, type: 'clarity' },
    ];
    const documentFor = (opacity: number, x: number) =>
      docWith([
        groupContract(
          'g',
          [
            raster('b', { opacity, transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y: 15 } }),
            raster('m', { transform: translated }),
          ],
          { adjustments: stack }
        ),
      ]);
    const display = () => {
      const diagnostics = createCanvasDiagnostics(true);
      const groupSurfaces = createGroupSurfaceCache({
        createSurface: (w, h) => scene.backend.createSurface(w, h),
        damageSince: (id, version) => scene.caches.damageSince(id, version),
        diagnostics,
        getAdjustedSurface: RAW_MEMBERS,
        getCacheEntry: (id) => scene.caches.get(id),
        spatial: { run: (request) => service.run(request) },
      });
      const screen = scene.backend.createSurface(WIDTH, HEIGHT);
      let state: GroupSurfaceResult['state'] | undefined;
      const show = (document: ReturnType<typeof documentFor>) =>
        settle(
          () => {
            screen.ctx.clearRect(0, 0, WIDTH, HEIGHT);
            compositeDocument(screen, document, scene.caches, identity(), {
              backend: scene.backend,
              groupSurface: (scope, members, matrices, content) => {
                const result = groupSurfaces.get(scope, members, matrices, content);
                state = result?.state;
                return result;
              },
            });
          },
          () => state === 'exact'
        );
      return { diagnostics, screen, show };
    };

    const edited = display();
    await edited.show(documentFor(1, 20));
    await edited.show(documentFor(0.5, 23));
    expect(edited.diagnostics.snapshot()).toMatchObject({ groupSurfaceAllocations: 1 });
    expect(edited.diagnostics.snapshot().groupSurfaceRefreshes).toBeGreaterThan(0);

    const fresh = display();
    await fresh.show(documentFor(0.5, 23));
    expect(worstDifference(pixelsOf(edited.screen), pixelsOf(fresh.screen))).toBe(0);
  });
});
