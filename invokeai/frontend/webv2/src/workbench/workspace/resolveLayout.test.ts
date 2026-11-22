import type { WidgetRegion } from '@workbench/layoutContracts';

import { getRegionOrder, writeRegionOrder } from '@workbench/floatingWindows';
import { WIDGET_REGIONS } from '@workbench/layoutContracts';
import { getLayoutPreset } from '@workbench/layoutPresets';
import { describe, expect, it } from 'vitest';

import type { LayoutAvailability, ResolvedLayout } from './resolveLayout';
import type { WorkspaceStructure } from './structure';

import {
  createRandom,
  generateAvailability,
  generateStructure,
  generateView,
  registryAvailability,
  withUnavailable,
} from './layoutFixtures.testing';
import { createLayoutResolver, resolveLayout } from './resolveLayout';
import { normalizeStructure, PANEL_OPEN_KEYS } from './structure';
import { reduceStructure } from './structureReducer';
import { frontRegion, setRegionSize, setWindowGeometry } from './view';

const structureOf = (presetId: string) => normalizeStructure(getLayoutPreset(presetId).snapshot);
const edit = structureOf('edit');
const compose = structureOf('compose');

/** The resolved layout read back as a structure, so resolving it again must change nothing. */
const asStructure = (layout: ResolvedLayout, structure: WorkspaceStructure): WorkspaceStructure => ({
  ...(layout.floatingWidgets ? { floatingWidgets: layout.floatingWidgets } : {}),
  layout: { ...structure.layout, panels: layout.panels },
  widgetInstances: layout.widgetInstances,
  widgetRegions: layout.widgetRegions,
});

const arrangementOf = ({ floatingWidgets, panels, widgetRegions }: ResolvedLayout) => ({
  floatingWidgets,
  panels,
  widgetRegions,
});

const expectValid = (layout: ResolvedLayout, availability: LayoutAvailability) => {
  const typeOf = (instanceId: string) => layout.widgetInstances[instanceId]?.typeId ?? '';
  const windows = layout.floatingWidgets ?? {};
  const centerFront = layout.widgetRegions.center.activeInstanceId;
  const sideFronts = WIDGET_REGIONS.filter((region) => region !== 'center').map(
    (region) => layout.widgetRegions[region].activeInstanceId
  );
  // A center that yields a shared instance to a panel has nothing else it could front.
  const centerCanYield = layout.widgetRegions.center.instanceIds.some(
    (instanceId) =>
      instanceId !== centerFront &&
      !sideFronts.includes(instanceId) &&
      availability.getPresentation(typeOf(instanceId), 'center') === 'view'
  );
  let openSharers = 0;

  for (const region of WIDGET_REGIONS) {
    const { activeInstanceId, instanceIds, isCollapsed, sizePx } = layout.widgetRegions[region];

    expect(new Set(instanceIds).size).toBe(instanceIds.length);
    for (const instanceId of instanceIds) {
      expect(Object.hasOwn(windows, instanceId)).toBe(false);
      expect(availability.getPresentation(typeOf(instanceId), region)).not.toBeNull();
    }
    // Every front is a member, except the empty center that keeps naming what it lost.
    expect(
      instanceIds.includes(activeInstanceId) ||
        (instanceIds.length === 0 && (region === 'center' || activeInstanceId === ''))
    ).toBe(true);

    if (region === 'center') {
      expect(isCollapsed).toBe(false);
      continue;
    }

    expect(layout.panels[PANEL_OPEN_KEYS[region]]).toBe(!isCollapsed);
    expect(sizePx).toBeGreaterThanOrEqual(region === 'bottom' ? 96 : 350);
    expect(sizePx).toBeLessThanOrEqual(region === 'bottom' ? 420 : 720);

    if (activeInstanceId === centerFront && availability.getPresentation(typeOf(centerFront), region) === 'view') {
      const isDistinct =
        availability.hasDistinctSurface(typeOf(centerFront), 'center') &&
        availability.hasDistinctSurface(typeOf(centerFront), region);

      if (!isDistinct && !isCollapsed) {
        openSharers += 1;
        expect(centerCanYield).toBe(false);
      }
    }
  }

  expect(openSharers).toBeLessThanOrEqual(1);

  for (const [instanceId, window] of Object.entries(windows)) {
    expect(availability.getPresentation(typeOf(instanceId), window.returnRegion)).not.toBeNull();
    expect(window.widthPx).toBeGreaterThanOrEqual(280);
    expect(window.heightPx).toBeGreaterThanOrEqual(200);
  }

  // Return markers are canonical: each window's index is its slot among what its region shows.
  for (const region of WIDGET_REGIONS) {
    const order = writeRegionOrder(
      getRegionOrder(region, layout.widgetRegions[region].instanceIds, layout.floatingWidgets),
      layout.floatingWidgets
    );

    expect(order.floatingWidgets).toBe(layout.floatingWidgets);
  }
};

describe('resolveLayout repairs', () => {
  it('1: skips an instance whose type is unavailable, window included, and leaves the structure alone', () => {
    const floated = reduceStructure(compose, { instanceId: 'queue', type: 'floatWidget' }, registryAvailability);
    const before = structuredClone(floated);
    const layout = resolveLayout(floated, undefined, withUnavailable(registryAvailability, ['image-map', 'queue']));

    expect(layout.widgetRegions.right.instanceIds).toEqual(['gallery']);
    expect(layout.floatingWidgets).toBeUndefined();
    expect(floated).toEqual(before);
  });

  it('2: a front that is no available member falls back to the home front, then the first available member', () => {
    const stale = resolveLayout(edit, { fronts: { left: 'ghost', right: 'generate' } }, registryAvailability);

    expect(stale.widgetRegions.left.activeInstanceId).toBe('generate');
    expect(stale.widgetRegions.right.activeInstanceId).toBe('layers');

    const unavailableHome = resolveLayout(edit, undefined, withUnavailable(registryAvailability, ['layers']));

    expect(unavailableHome.widgetRegions.right.activeInstanceId).toBe('preview');
  });

  it('3: a center with no available view shows the default stage as a guest', () => {
    const canvasOnly = reduceStructure(
      edit,
      { instanceId: 'preview', region: 'center', type: 'removeWidget' },
      registryAvailability
    );
    const layout = resolveLayout(canvasOnly, undefined, withUnavailable(registryAvailability, ['canvas']));

    expect(layout.widgetRegions.center).toMatchObject({ activeInstanceId: 'preview', instanceIds: ['preview'] });
    expect(layout.guests).toEqual(['preview']);
  });

  it('4: an instance the center fronts is shown once, unless its type has a surface in both regions', () => {
    const away = resolveLayout(edit, { fronts: { center: 'preview', right: 'preview' } }, registryAvailability);

    expect(away.widgetRegions.right).toMatchObject({ activeInstanceId: 'layers', isCollapsed: false });

    // Both home fronts: the stage keeps it. An explicit rail front wins, and the center fronts its next view.
    const sharedHome = normalizeStructure({
      ...edit,
      widgetRegions: {
        ...edit.widgetRegions,
        center: { ...edit.widgetRegions.center, activeInstanceId: 'preview' },
        right: { ...edit.widgetRegions.right, activeInstanceId: 'preview' },
      },
    });
    const stageWins = resolveLayout(sharedHome, undefined, registryAvailability);
    const panel = resolveLayout(
      sharedHome,
      frontRegion(undefined, sharedHome, 'right', 'preview', registryAvailability),
      registryAvailability
    );

    expect(stageWins.widgetRegions.center.activeInstanceId).toBe('preview');
    expect(stageWins.widgetRegions.right.activeInstanceId).toBe('layers');
    expect(panel.widgetRegions.right.activeInstanceId).toBe('preview');
    expect(panel.widgetRegions.center.activeInstanceId).toBe('canvas');

    const video = structureOf('video');
    const railed = reduceStructure(
      video,
      { createNew: false, mintId: 'preview', region: 'right', type: 'addWidget', typeId: 'preview' },
      registryAvailability
    );
    const onlyView = resolveLayout(railed, { fronts: { right: 'preview' } }, registryAvailability);

    // A center with no other view shows it too.
    expect(onlyView.widgetRegions.right.activeInstanceId).toBe('preview');
    expect(onlyView.widgetRegions.center.activeInstanceId).toBe('preview');
    // A closed panel does not take it.
    expect(
      resolveLayout(railed, { collapsed: { right: true }, fronts: { right: 'preview' } }, registryAvailability)
        .widgetRegions.right.activeInstanceId
    ).toBe('gallery');

    const add = (structure: WorkspaceStructure, region: WidgetRegion) =>
      reduceStructure(
        structure,
        { mintId: 'flow', region, type: 'addWidget', typeId: 'workflow' },
        registryAvailability
      );
    const shared = add(add(compose, 'left'), 'center');
    const graphAndForm = resolveLayout(shared, { fronts: { center: 'flow', left: 'flow' } }, registryAvailability);

    expect(shared.widgetRegions.center.instanceIds).toContain('flow');
    expect(graphAndForm.widgetRegions.left.activeInstanceId).toBe('flow');
    expect(graphAndForm.widgetRegions.center.activeInstanceId).toBe('flow');
  });

  it('5: clamps sizes and window sizes to their limits', () => {
    const floated = reduceStructure(compose, { instanceId: 'queue', type: 'floatWidget' }, registryAvailability);
    const view = { sizes: { left: 2000 }, windows: { queue: { heightPx: 10, widthPx: 10 } } };
    const layout = resolveLayout(floated, view, registryAvailability);

    expect(layout.widgetRegions.left.sizePx).toBe(720);
    expect(layout.floatingWidgets?.queue).toMatchObject({ heightPx: 200, widthPx: 280 });
  });
});

describe('resolveLayout over generated inputs', () => {
  it('is valid, deterministic, and a fixed point of its own output', () => {
    for (let seed = 1; seed <= 300; seed += 1) {
      const random = createRandom(seed);
      const structure = generateStructure(random, registryAvailability, random.int(14));
      const view = generateView(random, structure);
      const availability = random.int(2) ? generateAvailability(random, structure) : registryAvailability;
      const layout = resolveLayout(structure, view, availability);

      expectValid(layout, availability);
      expect(resolveLayout(structure, view, availability)).toEqual(layout);
      expect(arrangementOf(resolveLayout(asStructure(layout, structure), undefined, availability))).toEqual(
        arrangementOf(layout)
      );
    }
  });
});

describe('createLayoutResolver', () => {
  it('keeps every part a view change did not touch, and the whole layout for the same inputs', () => {
    const floated = reduceStructure(compose, { instanceId: 'queue', type: 'floatWidget' }, registryAvailability);
    const resolve = createLayoutResolver();
    const view = frontRegion(
      setRegionSize(undefined, floated, 'right', 500),
      floated,
      'left',
      'upscale',
      registryAvailability
    );
    const first = resolve(floated, view, registryAvailability);

    expect(resolve(floated, view, registryAvailability)).toBe(first);

    const resizedView = setRegionSize(view, floated, 'left', 600);
    const resized = resolve(floated, resizedView, registryAvailability);

    expect(resized.widgetRegions.left).not.toBe(first.widgetRegions.left);
    for (const region of ['right', 'bottom', 'center'] as const) {
      expect(resized.widgetRegions[region]).toBe(first.widgetRegions[region]);
    }
    expect(resized.floatingWidgets).toBe(first.floatingWidgets);
    expect(resized.panels).toBe(first.panels);

    const geometry = { heightPx: 300, widthPx: 300, x: 5, y: 5 };
    const moved = resolve(floated, setWindowGeometry(resizedView, floated, 'queue', geometry), registryAvailability);

    expect(moved.floatingWidgets).not.toBe(resized.floatingWidgets);
    expect(moved.widgetRegions).toBe(resized.widgetRegions);
  });
});
