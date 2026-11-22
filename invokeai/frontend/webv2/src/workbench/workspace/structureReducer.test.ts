import { getLayoutPreset } from '@workbench/layoutPresets';
import { describe, expect, it } from 'vitest';

import type { WorkspaceStructure } from './structure';
import type { StructureAction } from './structureReducer';

import { createRandom, randomStructureAction, registryAvailability, withUnavailable } from './layoutFixtures.testing';
import { resolveLayout } from './resolveLayout';
import { normalizeStructure } from './structure';
import { captureLayout, reduceStructure, takesLastCenterView } from './structureReducer';
import { frontRegion, pruneView, setRegionCollapsed, setRegionSize, setWindowGeometry } from './view';

const structureOf = (presetId: string) => normalizeStructure(getLayoutPreset(presetId).snapshot);
const compose = structureOf('compose');
const edit = structureOf('edit');
const reduce = (structure: WorkspaceStructure, ...actions: StructureAction[]) =>
  actions.reduce((next, action) => reduceStructure(next, action, registryAvailability), structure);
const add = (region: 'left' | 'right' | 'bottom' | 'center', typeId: string, extra: Partial<StructureAction> = {}) =>
  ({ mintId: `new-${typeId}`, region, type: 'addWidget', typeId, ...extra }) as StructureAction;

describe('reduceStructure', () => {
  it('adds the instance already placed, else the project’s unplaced one, else a new one, at a docked index', () => {
    expect(reduce(compose, add('right', 'preview', { index: 1 })).widgetRegions.right.instanceIds).toEqual([
      'gallery',
      'preview',
      'image-map',
      'queue',
    ]);
    expect(reduce(compose, add('right', 'layers', { unplacedInstanceId: 'layers' })).widgetInstances.layers).toEqual({
      id: 'layers',
      typeId: 'layers',
    });
    expect(reduce(compose, add('right', 'queue', { createNew: true })).widgetRegions.right.instanceIds).toEqual([
      'gallery',
      'image-map',
      'queue',
      'new-queue',
    ]);
    expect(reduce(compose, add('left', 'layers'))).toBe(compose);
  });

  it('removes a member, fronting the next one and dropping an instance nothing places', () => {
    const removed = reduce(compose, { instanceId: 'gallery', region: 'right', type: 'removeWidget' });

    expect(removed.widgetRegions.right).toMatchObject({
      activeInstanceId: 'image-map',
      instanceIds: ['image-map', 'queue'],
    });
    expect(removed.widgetInstances.gallery).toBeUndefined();

    const emptied = reduce(
      edit,
      ...['layers', 'preview'].map((instanceId) => ({ instanceId, region: 'right', type: 'removeWidget' }) as const)
    );

    expect(emptied.widgetRegions.right).toMatchObject({ activeInstanceId: '', instanceIds: [], isCollapsed: true });
    expect(emptied.layout.panels.isRightOpen).toBe(false);
  });

  it('moves a tab between regions around window markers, opening a rail it gives its first member', () => {
    const floated = reduce(compose, { instanceId: 'generate', type: 'floatWidget' });
    const moved = reduce(floated, {
      fromRegion: 'right',
      instanceId: 'image-map',
      toIndex: 0,
      toRegion: 'left',
      type: 'moveWidget',
    });

    expect(moved.widgetRegions.right).toMatchObject({ activeInstanceId: 'gallery', instanceIds: ['gallery', 'queue'] });
    // `toIndex` counts docked members: the tab lands before Upscale, behind Generate's marker.
    expect(moved.widgetRegions.left.instanceIds).toEqual(['image-map', 'upscale']);
    expect(moved.floatingWidgets?.generate?.returnIndex).toBe(0);

    const emptied = reduce(
      compose,
      ...['gallery', 'image-map', 'queue'].map(
        (instanceId) => ({ instanceId, region: 'right', type: 'removeWidget' }) as const
      )
    );
    const refilled = reduce(emptied, {
      fromRegion: 'center',
      instanceId: 'gallery:center',
      toIndex: 0,
      toRegion: 'right',
      type: 'moveWidget',
    });

    expect(emptied.widgetRegions.right.isCollapsed).toBe(true);
    expect(refilled.widgetRegions.right).toMatchObject({ activeInstanceId: 'gallery:center', isCollapsed: false });
    expect(
      reduce(compose, { fromRegion: 'left', instanceId: 'upscale', toIndex: 0, toRegion: 'right', type: 'moveWidget' })
    ).toBe(compose);
  });

  it('reorders only by a permutation of docked members', () => {
    expect(
      reduce(compose, { instanceIds: ['queue', 'gallery', 'image-map'], region: 'right', type: 'reorderWidgets' })
        .widgetRegions.right.instanceIds
    ).toEqual(['queue', 'gallery', 'image-map']);
    expect(reduce(compose, { instanceIds: ['queue', 'gallery'], region: 'right', type: 'reorderWidgets' })).toBe(
      compose
    );
  });

  it('moves a status item between the bar’s clusters', () => {
    const aligned = reduce(compose, {
      align: 'end',
      instanceId: 'server-status',
      region: 'bottom',
      type: 'setAlignment',
    });

    expect(aligned.widgetRegions.bottom.alignEndInstanceIds).toEqual(['shortcuts', 'notifications', 'server-status']);
    expect(
      reduce(aligned, { align: 'start', instanceId: 'server-status', region: 'bottom', type: 'setAlignment' })
    ).toEqual(compose);
  });

  it('floats a widget out of every region and docks it back to each slot it left; closing removes it', () => {
    const shared = reduce(compose, add('right', 'preview', { index: 1 }));
    const floated = reduce(shared, {
      geometry: { heightPx: 300, widthPx: 400, x: 10, y: 20 },
      instanceId: 'preview',
      region: 'right',
      type: 'floatWidget',
    });

    expect(floated.widgetRegions.center.instanceIds).toEqual(['gallery:center']);
    expect(floated.widgetRegions.right.instanceIds).toEqual(['gallery', 'image-map', 'queue']);
    expect(floated.floatingWidgets?.preview).toMatchObject({
      alsoIn: { center: { index: 0, isFront: true } },
      heightPx: 300,
      mode: 'windowed',
      returnIndex: 1,
      returnRegion: 'right',
    });
    // The center gets back the stage the float took from it, in front again.
    expect(reduce(floated, { instanceId: 'preview', type: 'dockWindow' }).widgetRegions).toEqual(shared.widgetRegions);

    const closed = reduce(floated, { instanceId: 'preview', type: 'closeWindow' });

    expect(closed.floatingWidgets).toBeUndefined();
    expect(closed.widgetInstances.preview).toBeUndefined();
  });

  it('never takes the center’s last view', () => {
    const canvasOnly = reduce(edit, { instanceId: 'preview', region: 'center', type: 'removeWidget' });

    expect(canvasOnly.widgetRegions.center.instanceIds).toEqual(['canvas']);
    for (const action of [
      { instanceId: 'canvas', region: 'center', type: 'removeWidget' },
      { fromRegion: 'center', instanceId: 'canvas', toIndex: 0, toRegion: 'right', type: 'moveWidget' },
      { instanceId: 'canvas', type: 'floatWidget' },
    ] as const) {
      expect(reduce(canvasOnly, action)).toBe(canvasOnly);
    }
    // The shell asks the same question: what a session cannot show does not count as a view.
    expect(takesLastCenterView(edit, 'canvas', registryAvailability)).toBe(false);
    expect(takesLastCenterView(edit, 'canvas', withUnavailable(registryAvailability, ['preview']))).toBe(true);
    // Preview unavailable, Canvas is the center's last view though not its last member.
    expect(
      reduceStructure(
        edit,
        { instanceId: 'canvas', region: 'center', type: 'removeWidget' },
        withUnavailable(registryAvailability, ['preview'])
      )
    ).toBe(edit);

    // A structure stored before that rule, with the center's last view floated away, loads with it docked.
    const floated: WorkspaceStructure = {
      ...canvasOnly,
      floatingWidgets: {
        canvas: {
          heightPx: 400,
          mode: 'windowed',
          returnIndex: 0,
          returnRegion: 'center',
          stackOrder: 1,
          widthPx: 600,
          x: 0,
          y: 0,
        },
      },
      widgetRegions: { ...canvasOnly.widgetRegions, center: { ...canvasOnly.widgetRegions.center, instanceIds: [] } },
    };

    expect(normalizeStructure(floated)).toEqual(canvasOnly);
  });

  it('reads a closed panel and a collapsed region as one fact', () => {
    const closedPanel = {
      ...compose,
      layout: { ...compose.layout, panels: { ...compose.layout.panels, isLeftOpen: false } },
    };

    expect(normalizeStructure(closedPanel).widgetRegions.left.isCollapsed).toBe(true);
  });

  it('captures what the view shows as the defaults, leaving the view nothing to override', () => {
    const floated = reduce(edit, {
      geometry: { heightPx: 300, widthPx: 400, x: 10, y: 10 },
      instanceId: 'layers',
      region: 'right',
      type: 'floatWidget',
    });
    const view = setWindowGeometry(
      setRegionCollapsed(
        setRegionSize(
          frontRegion(
            frontRegion(undefined, floated, 'center', 'preview', registryAvailability),
            floated,
            'left',
            'upscale',
            registryAvailability
          ),
          floated,
          'right',
          500
        ),
        floated,
        'left',
        true
      ),
      floated,
      'layers',
      { heightPx: 320, widthPx: 420, x: 50, y: 60 }
    );
    const captured = captureLayout(
      floated,
      view,
      resolveLayout(floated, view, registryAvailability),
      registryAvailability
    );

    expect(captured.widgetRegions.center.activeInstanceId).toBe('preview');
    expect(captured.layout.centerViewId).toBe('preview');
    expect(captured.widgetRegions.left).toMatchObject({ activeInstanceId: 'upscale', isCollapsed: true });
    expect(captured.widgetRegions.right.sizePx).toBe(500);
    expect(captured.floatingWidgets?.layers).toMatchObject({ heightPx: 320, widthPx: 420, x: 50, y: 60 });
    expect(pruneView(view, captured, registryAvailability)).toBeUndefined();
  });

  it('returns a canonical structure after any sequence, and the same object for an edit that changes nothing', () => {
    for (let seed = 1; seed <= 200; seed += 1) {
      const random = createRandom(seed);
      let structure = normalizeStructure(random.pick([compose, edit, structureOf('automate'), structureOf('video')]));

      for (let step = 0; step < 20; step += 1) {
        const next = reduceStructure(structure, randomStructureAction(random, structure, step), registryAvailability);

        expect(normalizeStructure(next)).toBe(next);
        expect(next.widgetRegions.center.instanceIds.length).toBeGreaterThan(0);
        expect(
          reduceStructure(
            next,
            { instanceIds: next.widgetRegions.center.instanceIds, region: 'center', type: 'reorderWidgets' },
            registryAvailability
          )
        ).toBe(next);
        structure = next;
      }
    }
  });
});
