import type { CanvasAdjustmentsContract, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { createCanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { identity } from '@workbench/canvas-engine/math/mat2d';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { createColorSampler } from '@workbench/canvas-engine/render/colorSample';
import { compositeDocument } from '@workbench/canvas-engine/render/compositor';
import { createGroupSurfaceCache } from '@workbench/canvas-engine/render/groupSurfaceCache';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { planBaseRasterComposite, renderRasterComposite } from '@workbench/canvas-engine/render/rasterComposite';
import { describe, expect, it } from 'vitest';

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

/** Midtone-lifting levels (gamma 2): deliberately NON-affine, so applying it to a
 * composite differs measurably from applying it to each member before blending. */
const gammaStack = (id: string): CanvasAdjustmentsContract => [
  { gamma: 2, id, inBlack: 0, inWhite: 255, isEnabled: true, outBlack: 0, outWhite: 255, type: 'levels' },
];
const invertStack = (id: string): CanvasAdjustmentsContract => [{ id, isEnabled: true, type: 'invert' }];

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
    // gamma-2 on the COMPOSITE lifts g/b to ~180. Per-member baking would
    // leave g/b near ~127 (gamma fixes 0 and 255), 50+ levels away.
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

  it('applies nested stacks inner-first', async () => {
    // Use gray 64 to distinguish nested order: invert then gamma-2 yields about 221; reversed order yields about
    // 127.
    const document = docWith([
      groupContract('outer', [groupContract('inner', [raster('gray')], { adjustments: invertStack('ia') })], {
        adjustments: gammaStack('oa'),
      }),
    ]);
    const scene = sceneFor({ gray: '#404040' });
    const surface = await renderRasterComposite(planBaseRasterComposite(document, BBOX), scene);
    const [r] = centerPixel(surface);
    expect(Math.abs(r! - 221)).toBeLessThanOrEqual(3);
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
      getAdjustedSurface: () => null,
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
    // And a second composite reuses the cached group surface (same pixels).
    const again = scene.backend.createSurface(WIDTH, HEIGHT);
    compositeDocument(again, document, scene.caches, identity(), {
      backend: scene.backend,
      groupSurface: (scope, members, matrices, content) => groupSurfaces.get(scope, members, matrices, content),
    });
    expect(centerPixel(again)).toEqual(screenPx);
  });

  it('resolves sibling scopes nested inside a parent scope with correct member indexing', async () => {
    // Outer gamma-2 maps inverted `a` from 64 to about 221 and plain `b` to about 128; `d` does not cover center.
    const document = docWith([
      groupContract(
        'outer',
        [
          groupContract('i1', [raster('a')], { adjustments: invertStack('ia') }),
          raster('b', { opacity: 0.0 }),
          groupContract('i2', [raster('d', { opacity: 0.0 })], { adjustments: invertStack('ib') }),
        ],
        { adjustments: gammaStack('oa') }
      ),
    ]);
    const scene = sceneFor({ a: '#404040', b: '#404040', d: '#404040' });
    const exported = await renderRasterComposite(planBaseRasterComposite(document, BBOX), scene);
    const [r] = centerPixel(exported);
    expect(Math.abs(r! - 221)).toBeLessThanOrEqual(3);

    const groupSurfaces = createGroupSurfaceCache({
      createSurface: (w, h) => scene.backend.createSurface(w, h),
      getAdjustedSurface: () => null,
      damageSince: (id, version) => scene.caches.damageSince(id, version),
      getCacheEntry: (id) => scene.caches.get(id),
    });
    const screen = scene.backend.createSurface(WIDTH, HEIGHT);
    compositeDocument(screen, document, scene.caches, identity(), {
      backend: scene.backend,
      groupSurface: (scope, members, matrices, content) => groupSurfaces.get(scope, members, matrices, content),
    });
    expect(Math.abs(centerPixel(screen)[0]! - 221)).toBeLessThanOrEqual(3);
  });

  it('color-samples through a group stack when the providers are wired', () => {
    const document = docWith([groupContract('g', [raster('gray')], { adjustments: invertStack('ia') })]);
    const scene = sceneFor({ gray: '#404040' });
    const groupSurfaces = createGroupSurfaceCache({
      createSurface: (w, h) => scene.backend.createSurface(w, h),
      getAdjustedSurface: () => null,
      damageSince: (id, version) => scene.caches.damageSince(id, version),
      getCacheEntry: (id) => scene.caches.get(id),
    });
    const point = { x: WIDTH / 2, y: HEIGHT / 2 };
    const sampler = createColorSampler(scene.backend);
    const raw = sampler.sample(document, scene.caches, point);
    expect(raw?.r).toBe(0x40);
    const adjusted = sampler.sample(document, scene.caches, point, {
      groupSurface: (scope, members, matrices, content) => groupSurfaces.get(scope, members, matrices, content),
    });
    expect(adjusted?.r).toBe(255 - 0x40);
  });

  it('applies group opacity to the isolated composite, matching the export renderer', async () => {
    // Opaque red over opaque white inside a 50%-opacity group, over a blue
    // base: the GROUP composite is pure red (white hidden underneath), so the
    // result is red 50% over blue. Per-leaf opacity instead would let white
    // bleed through inside the group.
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
      getAdjustedSurface: () => null,
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

  it('applies group blend mode when the isolated composite lands, on screen and in export', async () => {
    // Nesting the multiply group proves its composite lands inside the parent, not only at the root.
    const document = docWith([groupContract('g', [raster('gray')], { blendMode: 'multiply' }), raster('under')]);
    const scene = sceneFor({ gray: '#808080', under: '#ffffff' });
    const exported = await renderRasterComposite(planBaseRasterComposite(document, BBOX), scene);
    expect(Math.abs(centerPixel(exported)[0]! - 0x80)).toBeLessThanOrEqual(1);

    const nested = docWith([
      groupContract('outer', [groupContract('inner', [raster('gray')], { blendMode: 'multiply' }), raster('mid')], {
        adjustments: invertStack('oa'),
      }),
    ]);
    const nestedScene = sceneFor({ gray: '#808080', mid: '#c0c0c0' });
    // inner multiplies onto mid inside outer's buffer: 0x80×0xc0/0xff ≈ 0x60;
    // outer's invert flips it to ≈ 0x9f.
    const nestedExport = await renderRasterComposite(planBaseRasterComposite(nested, BBOX), nestedScene);
    expect(Math.abs(centerPixel(nestedExport)[0]! - (255 - 0x60))).toBeLessThanOrEqual(3);

    const groupSurfaces = createGroupSurfaceCache({
      createSurface: (w, h) => nestedScene.backend.createSurface(w, h),
      getAdjustedSurface: () => null,
      damageSince: (id, version) => nestedScene.caches.damageSince(id, version),
      getCacheEntry: (id) => nestedScene.caches.get(id),
    });
    const screen = nestedScene.backend.createSurface(WIDTH, HEIGHT);
    compositeDocument(screen, nested, nestedScene.caches, identity(), {
      backend: nestedScene.backend,
      groupSurface: (scope, members, matrices, content) => groupSurfaces.get(scope, members, matrices, content),
    });
    const screenPx = centerPixel(screen);
    const exportPx = centerPixel(nestedExport);
    for (let channel = 0; channel < 4; channel += 1) {
      expect(Math.abs(screenPx[channel]! - exportPx[channel]!)).toBeLessThanOrEqual(1);
    }
  });

  it('reuses the cached group surface across an opacity scrub (only the landing changes)', () => {
    const scene = sceneFor({ red: '#ff0000' });
    let builds = 0;
    const groupSurfaces = createGroupSurfaceCache({
      createSurface: (w, h) => {
        builds += 1;
        return scene.backend.createSurface(w, h);
      },
      getAdjustedSurface: () => null,
      damageSince: (id, version) => scene.caches.damageSince(id, version),
      getCacheEntry: (id) => scene.caches.get(id),
    });
    const compositeAt = (opacity: number) => {
      const document = docWith([groupContract('g', [raster('red')], { opacity })]);
      const screen = scene.backend.createSurface(WIDTH, HEIGHT);
      compositeDocument(screen, document, scene.caches, identity(), {
        backend: scene.backend,
        groupSurface: (scope, members, matrices, content) => groupSurfaces.get(scope, members, matrices, content),
      });
      return centerPixel(screen);
    };
    const at30 = compositeAt(0.3);
    const at60 = compositeAt(0.6);
    const at90 = compositeAt(0.9);
    expect(builds).toBe(1);
    expect(Math.abs(at30[3]! - 77)).toBeLessThanOrEqual(2);
    expect(Math.abs(at60[3]! - 153)).toBeLessThanOrEqual(2);
    expect(Math.abs(at90[3]! - 230)).toBeLessThanOrEqual(2);
  });

  it('holds two keys per group so alternating consumers (frame vs overview) stop redrawing', () => {
    // Frame/session and settled Overview matrices alternate two keys; two slots retain them, while a third
    // repaints the least-recently-used slot in place instead of allocating.
    const scene = sceneFor({ red: '#ff0000' });
    const diagnostics = createCanvasDiagnostics(true);
    const groupSurfaces = createGroupSurfaceCache({
      createSurface: (w, h) => scene.backend.createSurface(w, h),
      damageSince: (id, version) => scene.caches.damageSince(id, version),
      diagnostics,
      getAdjustedSurface: () => null,
      getCacheEntry: (id) => scene.caches.get(id),
    });
    const docAt = (x: number) =>
      docWith([
        groupContract('g', [raster('red', { transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y: 0 } })], {
          adjustments: invertStack('ia'),
        }),
      ]);
    const composite = (x: number) => {
      const screen = scene.backend.createSurface(WIDTH, HEIGHT);
      compositeDocument(screen, docAt(x), scene.caches, identity(), {
        backend: scene.backend,
        groupSurface: (scope, members, matrices, content) => groupSurfaces.get(scope, members, matrices, content),
      });
      return screen;
    };
    const draws = () => {
      const { groupSurfaceAllocations, groupSurfaceRebuilds } = diagnostics.snapshot();
      return { allocations: groupSurfaceAllocations, rebuilds: groupSurfaceRebuilds };
    };

    const first = centerPixel(composite(0));
    composite(1);
    expect(draws()).toEqual({ allocations: 2, rebuilds: 0 });
    composite(0);
    composite(1);
    composite(0);
    expect(draws()).toEqual({ allocations: 2, rebuilds: 0 });
    expect(centerPixel(composite(0))).toEqual(first);

    composite(2);
    expect(draws()).toEqual({ allocations: 2, rebuilds: 1 });
    // x:2 took over the least-recently-used slot (x:1); x:0 is still warm.
    composite(0);
    expect(draws()).toEqual({ allocations: 2, rebuilds: 1 });
    composite(1);
    expect(draws()).toEqual({ allocations: 2, rebuilds: 2 });

    // A null build (every member excluded) must not evict the warm slots the other consumer still needs.
    const screen = scene.backend.createSurface(WIDTH, HEIGHT);
    compositeDocument(screen, docAt(1), scene.caches, identity(), {
      backend: scene.backend,
      groupSurface: (scope, members, matrices) =>
        groupSurfaces.get(scope, members, matrices, { excludeIds: new Set(['red']), float: null, previews: null }),
    });
    composite(1);
    expect(draws()).toEqual({ allocations: 2, rebuilds: 2 });
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
    /** Member cache extent in source pixels; the transform maps it into the document. */
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
      getAdjustedSurface: () => null,
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
    /** Paints a magenta dot into the member cache and publishes only that damage. */
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

    // Green and red at the group's 50% over blue: the preview replaces the member, never the group's opacity.
    expect(committed[1]).toBeGreaterThan(120);
    expect(committed[1]).toBeLessThan(135);
    expect(previewed[0]).toBeGreaterThan(120);
    expect(previewed[0]).toBeLessThan(135);
    expect(previewed[2]).toBeGreaterThan(120);
    expect(previewed[3]).toBe(255);
  });

  it('draws a floating selection inside its group, under the group opacity', () => {
    const { composite, scene } = groupedScene();
    const lifted = scene.backend.createSurface(8, 8);
    lifted.ctx.fillStyle = '#ffffff';
    lifted.ctx.fillRect(0, 0, 8, 8);

    const screen = composite({
      floatingSelection: {
        layerId: 'member',
        matrix: identity(),
        rect: { height: 8, width: 8, x: WIDTH / 2 - 4, y: HEIGHT / 2 - 4 },
        surface: lifted,
      },
    });

    // White at 50% over blue, not opaque white drawn above the group.
    const [r, g, b] = centerPixel(screen);
    expect(Math.abs(r! - 128)).toBeLessThanOrEqual(2);
    expect(Math.abs(g! - 128)).toBeLessThanOrEqual(2);
    expect(Math.abs(b! - 255)).toBeLessThanOrEqual(2);
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

  it.each<[string, GroupedSceneOptions]>([
    ['an untransformed member', {}],
    [
      'a member scaled 4x',
      {
        member: { transform: { rotation: 0, scaleX: 4, scaleY: 4, x: 0, y: 0 } },
        memberRect: { height: 16, width: 16, x: 0, y: 0 },
      },
    ],
  ])('moves a floating selection on %s by refreshing only the old and new landing', (_, options) => {
    const lifted = (scene: ReturnType<typeof groupedScene>['scene']) => {
      const surface = scene.backend.createSurface(3, 3);
      surface.ctx.fillStyle = '#ffffff';
      surface.ctx.fillRect(0, 0, 3, 3);
      return surface;
    };
    const floatAt = (surface: RasterSurface, x: number) => ({
      floatingSelection: { layerId: 'member', matrix: identity(), rect: { height: 3, width: 3, x, y: 5 }, surface },
    });
    const dragged = groupedScene(options);
    const surface = lifted(dragged.scene);
    dragged.composite(floatAt(surface, 2));
    const before = dragged.diagnostics.snapshot();
    const pixels = allPixels(dragged.composite(floatAt(surface, 8)));
    const after = dragged.diagnostics.snapshot();

    expect(after.groupSurfaceRebuilds).toBe(before.groupSurfaceRebuilds);
    expect(after.groupSurfaceRefreshes).toBe(before.groupSurfaceRefreshes + 1);
    const fresh = groupedScene(options);
    expect(pixels).toEqual(allPixels(fresh.composite(floatAt(lifted(fresh.scene), 8))));
  });
});

describe('layer adjustments in export', () => {
  /** A 16px gradient layer, scaled 3x and offset, with a non-linear stack. */
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

  it('allocates only the layer-local region the bbox can sample, not a bbox-sized intermediate', async () => {
    const { document, scene } = adjustedScene();
    const sizes: [number, number][] = [];
    const backend = {
      ...scene.backend,
      createSurface: (width: number, height: number) => {
        sizes.push([width, height]);
        return scene.backend.createSurface(width, height);
      },
    };
    const corner: Rect = { height: 8, width: 8, x: 5, y: 7 };
    await renderRasterComposite(planBaseRasterComposite(document, corner), { ...scene, backend });

    // The output surface plus a local copy covering about 8/3 source pixels (+1px resampling margin) a side.
    expect(sizes).toEqual([
      [8, 8],
      [4, 4],
    ]);
  });

  it('adjusts a downscaled layer after resampling, so the work never exceeds the output', async () => {
    const { scene } = adjustedScene();
    const shrunk = docWith([
      raster('graded', {
        adjustments: gammaStack('ga'),
        transform: { rotation: 0, scaleX: 0.25, scaleY: 0.25, x: 0, y: 0 },
      }),
    ]);
    const sizes: [number, number][] = [];
    const backend = {
      ...scene.backend,
      createSurface: (width: number, height: number) => {
        sizes.push([width, height]);
        return scene.backend.createSurface(width, height);
      },
    };
    await renderRasterComposite(planBaseRasterComposite(shrunk, { height: 2, width: 2, x: 0, y: 0 }), {
      ...scene,
      backend,
    });

    expect(sizes).toEqual([
      [2, 2],
      [2, 2],
    ]);
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
