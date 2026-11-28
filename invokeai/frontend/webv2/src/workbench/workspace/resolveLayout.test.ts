import type { WidgetRegion } from '@workbench/layoutContracts';

import { getRegionOrder, writeRegionOrder } from '@workbench/floatingWindows';
import { WIDGET_REGIONS } from '@workbench/layoutContracts';
import { getLayoutPreset } from '@workbench/layoutPresets';
import { describe, expect, it } from 'vitest';

import type { LayoutAvailability, ResolvedLayout } from './resolveLayout';
import type { WorkspaceStructure } from './structure';
import type { WorkspaceView } from './view';

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
import { frontRegion, getGuestWindowRegion, pruneView, setRegionSize, setWindowGeometry, showGuest } from './view';
import { isPlacedInstance } from './workspaceDerivations';

const structureOf = (presetId: string) => normalizeStructure(getLayoutPreset(presetId).snapshot);
const edit = structureOf('edit');
const compose = structureOf('compose');
const withoutType = (structure: WorkspaceStructure, typeId: string): WorkspaceStructure =>
  WIDGET_REGIONS.reduce(
    (next, region) =>
      next.widgetRegions[region].instanceIds
        .filter((instanceId) => next.widgetInstances[instanceId]?.typeId === typeId)
        .reduce(
          (current, instanceId) =>
            reduceStructure(current, { instanceId, region, type: 'removeWidget' }, registryAvailability),
          next
        ),
    structure
  );

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

const expectValid = (
  layout: ResolvedLayout,
  structure: WorkspaceStructure,
  view: WorkspaceView | undefined,
  availability: LayoutAvailability
) => {
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

  // A guest shows where it is: a center guest fronted there, at most one, and a guest window among the windows.
  const guests = Object.entries(layout.guests);
  const centerGuests = guests.filter(([, surface]) => surface === 'center').map(([instanceId]) => instanceId);

  expect(centerGuests.length).toBeLessThanOrEqual(1);
  expect(centerGuests.every((instanceId) => instanceId === centerFront)).toBe(true);
  const panelSurface = (typeId: string, region: WidgetRegion) =>
    availability.hasDistinctSurface(typeId, region) ? region : 'panel';
  // Whether the layout shows a placed instance of the type, in a rail or a window, on a guest window's surface.
  const showsSameSurface = (typeId: string) => {
    const own = panelSurface(typeId, getGuestWindowRegion(availability.getPresentation, typeId)!);
    const isPlaced = (id: string) => typeOf(id) === typeId && layout.guests[id] === undefined;

    return (
      WIDGET_REGIONS.some(
        (region) =>
          region !== 'center' &&
          layout.widgetRegions[region].instanceIds.some(isPlaced) &&
          panelSurface(typeId, region) === own
      ) ||
      Object.entries(windows).some(([id, window]) => isPlaced(id) && panelSurface(typeId, window.returnRegion) === own)
    );
  };

  for (const [instanceId, surface] of guests) {
    // A guest is the structure's instance only as the center's fallback stage, and never one the structure places.
    expect(Object.hasOwn(structure.widgetInstances, instanceId) && instanceId !== availability.fallbackStage?.id).toBe(
      false
    );
    expect(
      Object.hasOwn(structure.floatingWidgets ?? {}, instanceId) ||
        WIDGET_REGIONS.some((region) => structure.widgetRegions[region].instanceIds.includes(instanceId))
    ).toBe(false);

    if (surface !== 'window') {
      continue;
    }

    const typeId = typeOf(instanceId);

    expect(Object.hasOwn(windows, instanceId)).toBe(true);
    expect(WIDGET_REGIONS.some((region) => layout.widgetRegions[region].instanceIds.includes(instanceId))).toBe(false);
    expect(showsSameSurface(typeId)).toBe(false);
  }

  // Conversely, a well-formed guest window the layout shows nothing in place of is shown.
  for (const [instanceId, guest] of Object.entries(view?.guests ?? {})) {
    const isOnlyOfType = Object.values(view?.guests ?? {}).filter(({ typeId }) => typeId === guest.typeId).length === 1;

    if (
      guest.surface === 'window' &&
      isOnlyOfType &&
      !Object.hasOwn(structure.widgetInstances, instanceId) &&
      getGuestWindowRegion(availability.getPresentation, guest.typeId) !== null &&
      !showsSameSurface(guest.typeId)
    ) {
      expect(layout.guests[instanceId]).toBe('window');
    }
  }

  // Return markers are canonical: each placed window's index is its slot among what its region shows.
  const placedWindows = Object.fromEntries(
    Object.entries(windows).filter(([instanceId]) => layout.guests[instanceId] === undefined)
  );

  for (const region of WIDGET_REGIONS) {
    const order = writeRegionOrder(
      getRegionOrder(region, layout.widgetRegions[region].instanceIds, placedWindows),
      placedWindows
    );

    expect(order.floatingWidgets).toBe(placedWindows);
  }
};

const withoutGuests = (view: WorkspaceView | undefined): WorkspaceView | undefined => {
  const { guests: _guests, ...rest } = view ?? {};

  return view && rest;
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

  it('3: a center with no available view shows the default stage as a guest, never as an instance a rail places', () => {
    const canvasOnly = reduceStructure(
      edit,
      { instanceId: 'preview', region: 'center', type: 'removeWidget' },
      registryAvailability
    );
    const unavailable = withUnavailable(registryAvailability, ['canvas']);
    const layout = resolveLayout(canvasOnly, undefined, unavailable);

    // Edit's right rail places the stage's own instance, so the center's guest is another Preview.
    expect(canvasOnly.widgetRegions.right.instanceIds).toContain('preview');
    expect(layout.widgetRegions.center).toMatchObject({
      activeInstanceId: 'preview:guest',
      instanceIds: ['preview:guest'],
    });
    expect(layout.guests).toEqual({ 'preview:guest': 'center' });
    expect(isPlacedInstance(layout, 'preview')).toBe(true);

    const railless = withoutType(canvasOnly, 'preview');
    expect(resolveLayout(railless, undefined, unavailable).guests).toEqual({ preview: 'center' });

    // A guest window already showing that instance wins: nothing shows twice.
    const noPreview = withoutType(canvasOnly, 'preview');
    const window = showGuest(
      undefined,
      noPreview,
      'preview',
      { surface: 'window', typeId: 'preview' },
      { heightPx: 300, widthPx: 400, x: 0, y: 0 }
    );
    const withWindow = resolveLayout(noPreview, window, withUnavailable(registryAvailability, ['canvas']));

    expect(withWindow.guests).toEqual({ preview: 'window' });
    expect(withWindow.widgetRegions.center.instanceIds).not.toContain('preview');
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

  it('5: drops a guest the structure supersedes, by its id or by the same surface anywhere, and keeps it in the view only until pruned', () => {
    const geometry = { heightPx: 300, widthPx: 400, x: 40, y: 40 };
    const withoutGallery = withoutType(edit, 'gallery');
    const galleryGuest = showGuest(
      undefined,
      withoutGallery,
      'gallery',
      { surface: 'window', typeId: 'gallery' },
      geometry
    );
    const guested = resolveLayout(withoutGallery, galleryGuest, registryAvailability);

    expect(guested.guests).toEqual({ gallery: 'window' });
    expect(guested.floatingWidgets?.gallery).toMatchObject({ ...geometry, returnRegion: 'left' });

    // Placed in another rail, as another instance: the rail is what shows, and pruning forgets the guest.
    const inRightRail = reduceStructure(
      withoutGallery,
      { createNew: true, mintId: 'gallery:center', region: 'right', type: 'addWidget', typeId: 'gallery' },
      registryAvailability
    );

    expect(resolveLayout(inRightRail, galleryGuest, registryAvailability).guests).toEqual({});
    expect(pruneView(galleryGuest, inRightRail, registryAvailability)).toBeUndefined();

    // Its own id placed in the center: hidden at once, forgotten on pruning, so removing that view brings nothing back.
    const byId = reduceStructure(
      withoutGallery,
      { createNew: true, mintId: 'gallery', region: 'center', type: 'addWidget', typeId: 'gallery' },
      registryAvailability
    );
    const pruned = pruneView(galleryGuest, byId, registryAvailability);

    expect(resolveLayout(byId, galleryGuest, registryAvailability).floatingWidgets?.gallery).toBeUndefined();
    expect(pruned).toBeUndefined();
    expect(
      resolveLayout(
        reduceStructure(byId, { instanceId: 'gallery', region: 'center', type: 'removeWidget' }, registryAvailability),
        pruned,
        registryAvailability
      ).guests
    ).toEqual({});
    expect(
      resolveLayout(withoutGallery, galleryGuest, withUnavailable(registryAvailability, ['gallery'])).guests
    ).toEqual({});

    // A workflow's form in a window is not its graph in the status bar's panel.
    const graphBelow = reduceStructure(
      withoutType(edit, 'workflow'),
      { mintId: 'workflow', region: 'bottom', type: 'addWidget', typeId: 'workflow' },
      registryAvailability
    );
    const formGuest = showGuest(
      undefined,
      graphBelow,
      'workflow:guest',
      { surface: 'window', typeId: 'workflow' },
      geometry
    );

    expect(resolveLayout(graphBelow, formGuest, registryAvailability).guests).toEqual({ 'workflow:guest': 'window' });

    // A placement repair 1 hides (Layers stored in the left rail, where it cannot show) supersedes nothing.
    const hiddenLayers = normalizeStructure({
      ...withoutType(edit, 'layers'),
      widgetRegions: {
        ...withoutType(edit, 'layers').widgetRegions,
        left: {
          ...edit.widgetRegions.left,
          instanceIds: [...edit.widgetRegions.left.instanceIds, 'layers'],
        },
      },
      widgetInstances: { ...withoutType(edit, 'layers').widgetInstances, layers: { id: 'layers', typeId: 'layers' } },
    });
    const layersGuest = showGuest(
      undefined,
      hiddenLayers,
      'layers:guest',
      { surface: 'window', typeId: 'layers' },
      geometry
    );

    expect(resolveLayout(hiddenLayers, layersGuest, registryAvailability).guests).toEqual({ 'layers:guest': 'window' });
    expect(pruneView(layersGuest, hiddenLayers, registryAvailability)).toBe(layersGuest);

    // Placed in a rail only, Preview still shows as the center's guest.
    const railPreview = reduceStructure(
      edit,
      { instanceId: 'preview', region: 'center', type: 'removeWidget' },
      registryAvailability
    );
    const away = resolveLayout(
      railPreview,
      showGuest(undefined, railPreview, 'preview:guest', { surface: 'center', typeId: 'preview' }, geometry),
      registryAvailability
    );

    expect(railPreview.widgetRegions.right.instanceIds).toContain('preview');
    expect(away.widgetRegions.center.activeInstanceId).toBe('preview:guest');
    expect(away.guests).toEqual({ 'preview:guest': 'center' });
  });

  it('6: clamps sizes and window sizes to their limits', () => {
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
      const placed = resolveLayout(structure, withoutGuests(view), availability);

      expectValid(layout, structure, view, availability);
      expect(resolveLayout(structure, view, availability)).toEqual(layout);
      expect(resolveLayout(structure, view, availability, layout)).toBe(layout);
      // The view's fronts stay: an explicit side front of an instance the center shares is an ask no structure holds.
      const asks = view?.fronts ? { fronts: view.fronts } : undefined;

      expect(arrangementOf(resolveLayout(asStructure(placed, structure), asks, availability))).toEqual(
        arrangementOf(placed)
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
