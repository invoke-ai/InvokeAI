import { getLayoutPreset } from '@workbench/layoutPresets';
import { describe, expect, it } from 'vitest';

import type { WorkspaceStructure } from './structure';

import { registryAvailability } from './layoutFixtures.testing';
import { normalizeStructure } from './structure';
import { reduceStructure } from './structureReducer';
import {
  frontRegion,
  getGuestInstanceId,
  MAX_GUESTS,
  normalizeView,
  normalizeViews,
  pruneView,
  raiseWindow,
  setRegionCollapsed,
  setRegionSize,
  setWindowGeometry,
  showGuest,
} from './view';

const compose = normalizeStructure(getLayoutPreset('compose').snapshot);
const withWindows = ['queue', 'image-map'].reduce(
  (structure, instanceId) => reduceStructure(structure, { instanceId, type: 'floatWidget' }, registryAvailability),
  compose
);

describe('workspace views', () => {
  it('store only what differs from the structure, and never a front the region does not hold', () => {
    const fronted = frontRegion(undefined, compose, 'right', 'queue', registryAvailability);

    expect(fronted).toEqual({ fronts: { right: 'queue' } });
    expect(frontRegion(fronted, compose, 'right', 'queue', registryAvailability)).toBe(fronted);
    expect(frontRegion(fronted, compose, 'right', 'gallery', registryAvailability)).toBeUndefined();
    expect(frontRegion(undefined, compose, 'right', 'generate', registryAvailability)).toBeUndefined();
    // A size the limits clamp to the default is the default.
    expect(setRegionSize(undefined, compose, 'left', 450)).toBeUndefined();
    expect(setRegionSize(undefined, compose, 'left', 9000)).toEqual({ sizes: { left: 720 } });
    expect(
      setRegionCollapsed(setRegionCollapsed(undefined, compose, 'right', true), compose, 'right', false)
    ).toBeUndefined();
    expect(setRegionCollapsed(undefined, compose, 'bottom', true)).toBeUndefined();

    const queue = withWindows.floatingWidgets!.queue!;
    const moved = setWindowGeometry(undefined, withWindows, 'queue', { ...queue, x: queue.x + 40 });

    expect(moved).toEqual({ windows: { queue: { x: queue.x + 40 } } });
  });

  it('raise a window by compacting stack orders, overriding only those that change', () => {
    const raised = raiseWindow(undefined, withWindows, 'queue');

    expect(raised).toEqual({ windows: { 'image-map': { stackOrder: 1 }, queue: { stackOrder: 2 } } });
    expect(raiseWindow(raised, withWindows, 'queue')).toBe(raised);
  });

  it('read stored views as untrusted data, keeping only well-typed overrides', () => {
    const junk = [
      null,
      7,
      'view',
      [],
      { fronts: 'left' },
      { sizes: { center: 400, left: -1 } },
      { windows: { a: {} } },
    ];

    for (const value of junk) {
      expect(normalizeView(value)).toBeUndefined();
    }
    expect(
      normalizeView({
        collapsed: { left: 'yes', right: true },
        fronts: { center: 'canvas', middle: 'x', right: 3 },
        sizes: { bottom: Number.NaN, left: 500 },
        windows: { queue: { mode: 'floating', stackOrder: 2, x: '1' } },
      })
    ).toEqual({
      collapsed: { right: true },
      fronts: { center: 'canvas' },
      sizes: { left: 500 },
      windows: { queue: { stackOrder: 2 } },
    });
    expect(
      normalizeViews(
        { compose: { fronts: { left: 'upscale' } }, gone: { fronts: { left: 'x' } } },
        (id) => id === 'compose'
      )
    ).toEqual({
      compose: { fronts: { left: 'upscale' } },
    });
  });

  it('read guests back as stored, one per type and one in the center, at most the bound, junk dropped', () => {
    const geometry = { heightPx: 300, widthPx: 400, x: 20, y: 30 };
    const view = showGuest(
      showGuest(undefined, compose, 'workflow', { surface: 'center', typeId: 'workflow' }, geometry),
      compose,
      'diagnostics',
      { surface: 'window', typeId: 'diagnostics' },
      geometry
    );

    expect(normalizeView(JSON.parse(JSON.stringify(view)))).toEqual(view);
    expect(
      normalizeView({
        guests: {
          a: { surface: 'center', typeId: 'canvas' },
          b: { surface: 'center', typeId: 'workflow' },
          c: { surface: 'window', typeId: 'canvas' },
          d: { surface: 'dock', typeId: 'queue' },
          e: { surface: 'window' },
          '': { surface: 'window', typeId: 'queue' },
        },
      })
    ).toEqual({ guests: { a: { surface: 'center', typeId: 'canvas' } } });

    const many = Object.fromEntries(
      Array.from({ length: MAX_GUESTS + 5 }, (_, index) => [`g${index}`, { surface: 'window', typeId: `t${index}` }])
    );

    expect(Object.keys(normalizeView({ guests: many })?.guests ?? {})).toHaveLength(MAX_GUESTS);
  });

  it('make a guest one of the project’s own instances the workspace does not use, named after the type first', () => {
    const using = (...ids: string[]) =>
      ({ widgetInstances: Object.fromEntries(ids.map((id) => [id, { id, typeId: 'gallery' }])) }) as WorkspaceStructure;
    const owned = ['gallery:center', 'gallery', 'gallery:bottom'];

    expect(getGuestInstanceId(using('gallery:center'), 'gallery', owned)).toBe('gallery');
    expect(getGuestInstanceId(using('gallery'), 'gallery', owned)).toBe('gallery:bottom');
    expect(getGuestInstanceId(using(...owned), 'gallery', owned)).toBe('gallery:guest');
    expect(getGuestInstanceId(using(...owned, 'gallery:guest'), 'gallery', owned)).toBe('gallery:guest-2');
  });

  it('drop overrides a changed structure no longer has, or now has as its defaults', () => {
    const view = {
      collapsed: { right: true },
      fronts: { left: 'upscale' },
      windows: { 'image-map': { x: 3 }, queue: { x: 1 } },
    };

    expect(pruneView(view, withWindows, registryAvailability)).toBe(view);

    const changed = [
      { instanceId: 'queue', type: 'closeWindow' },
      { instanceId: 'upscale', region: 'left', type: 'removeWidget' },
    ] as const;
    const edited = changed.reduce(
      (structure, action) => reduceStructure(structure, action, registryAvailability),
      withWindows
    );
    const collapsedRight = {
      ...edited,
      layout: { ...edited.layout, panels: { ...edited.layout.panels, isRightOpen: false } },
      widgetRegions: { ...edited.widgetRegions, right: { ...edited.widgetRegions.right, isCollapsed: true } },
    };

    expect(pruneView(view, collapsedRight, registryAvailability)).toEqual({ windows: { 'image-map': { x: 3 } } });
  });
});
