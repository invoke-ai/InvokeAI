import { getLayoutPreset } from '@workbench/layoutPresets';
import { describe, expect, it } from 'vitest';

import { registryAvailability } from './layoutFixtures.testing';
import { normalizeStructure } from './structure';
import { reduceStructure } from './structureReducer';
import {
  frontRegion,
  normalizeView,
  normalizeViews,
  pruneView,
  raiseWindow,
  setRegionCollapsed,
  setRegionSize,
  setWindowGeometry,
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
