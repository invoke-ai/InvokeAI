import type { InvocationRoute } from '@workbench/invocationContracts';

import { getNaturalDestination } from '@workbench/graphWidgets';
import { getAvailableSourceIdForWidgetTypeId } from '@workbench/invocationSources';
import { getLayoutPreset } from '@workbench/layoutPresets';
import { describe, expect, it } from 'vitest';

import type { WorkspaceStructure } from './structure';
import type { WorkspaceView } from './view';
import type { SavedWorkspace } from './workspaceDerivations';

import { enterWorkspace } from './entry';
import { registryAvailability } from './layoutFixtures.testing';
import { resolveLayout } from './resolveLayout';
import { normalizeStructure } from './structure';
import { frontRegion, setRegionCollapsed } from './view';

const sources = { getNaturalDestination, getSourceId: getAvailableSourceIdForWidgetTypeId };
const enter = (
  workspace: SavedWorkspace & { snapshot: WorkspaceStructure },
  view: WorkspaceView | undefined,
  invocation = route()
) =>
  enterWorkspace({
    invocation,
    resolve: (candidate) => resolveLayout(workspace.snapshot, candidate, registryAvailability),
    sources,
    structure: workspace.snapshot,
    surfaces: registryAvailability,
    view,
    workspace,
  });
const route = (overrides: Partial<InvocationRoute> = {}): InvocationRoute => ({
  destination: 'gallery',
  destinationLocked: false,
  sourceId: 'generate',
  sourceLocked: false,
  ...overrides,
});

const edit = getLayoutPreset('edit');
const compose = getLayoutPreset('compose');
const video = getLayoutPreset('video');

describe('enterWorkspace', () => {
  it('ends Away: Edit left on Preview comes back to Canvas, aimed Canvas to Canvas', () => {
    const away = frontRegion(undefined, edit.snapshot, 'center', 'preview', registryAvailability);
    const entry = enter(edit, away);

    expect(entry).toEqual({ invocation: route({ destination: 'canvas', sourceId: 'canvas' }), view: undefined });
    expect(resolveLayout(edit.snapshot, entry.view, registryAvailability).widgetRegions.center.activeInstanceId).toBe(
      'canvas'
    );
  });

  it('brings the stage back to the center from a rail that was asked for its instance', () => {
    const snapshot = normalizeStructure({
      ...edit.snapshot,
      widgetRegions: {
        ...edit.snapshot.widgetRegions,
        center: { ...edit.snapshot.widgetRegions.center, activeInstanceId: 'preview' },
        right: { ...edit.snapshot.widgetRegions.right, activeInstanceId: 'preview' },
      },
    });
    const railed = frontRegion(undefined, snapshot, 'right', 'preview', registryAvailability);
    const shown = (view: WorkspaceView | undefined) =>
      resolveLayout(snapshot, view, registryAvailability).widgetRegions;

    expect(shown(railed).center.activeInstanceId).toBe('canvas');

    const entry = enter({ ...edit, snapshot }, railed);

    expect(shown(entry.view).center.activeInstanceId).toBe('preview');
    expect(shown(entry.view).right.activeInstanceId).toBe('layers');
  });

  it('keeps each locked half of the route and still restores the stage', () => {
    const entry = enter(
      edit,
      frontRegion(undefined, edit.snapshot, 'center', 'preview', registryAvailability),
      route({ sourceLocked: true })
    );

    expect(entry.view).toBeUndefined();
    expect(entry.invocation).toEqual(route({ destination: 'canvas', sourceLocked: true }));
  });

  it('fronts the job’s source in a collapsed panel without opening it, keeping the rest of the view', () => {
    const view = setRegionCollapsed(
      frontRegion(
        frontRegion(undefined, compose.snapshot, 'left', 'upscale', registryAvailability),
        compose.snapshot,
        'right',
        'queue',
        registryAvailability
      ),
      compose.snapshot,
      'left',
      true
    );

    expect(enter(compose, view, route({ sourceId: 'upscale' }))).toEqual({
      invocation: route(),
      view: { collapsed: { left: true }, fronts: { right: 'queue' } },
    });
  });

  it('re-fronts Video in the Video rail after Upscale was fronted there', () => {
    expect(
      enter(
        video,
        frontRegion(undefined, video.snapshot, 'left', 'upscale', registryAvailability),
        route({ sourceId: 'upscale' })
      )
    ).toEqual({ invocation: route({ sourceId: 'video' }), view: undefined });
  });

  it('leaves the route of a workspace with no source as it was', () => {
    const structure = normalizeStructure({
      layout: {
        centerViewId: 'gallery',
        panels: { isBottomOpen: false, isLeftOpen: false, isRightOpen: true },
        presetId: 'review',
      },
      widgetInstances: {
        gallery: { id: 'gallery', typeId: 'gallery' },
        'image-map': { id: 'image-map', typeId: 'image-map' },
      },
      widgetRegions: {
        bottom: { activeInstanceId: '', instanceIds: [], isCollapsed: true, sizePx: 96 },
        center: { activeInstanceId: 'image-map', instanceIds: ['image-map'], isCollapsed: false, sizePx: 0 },
        left: { activeInstanceId: '', instanceIds: [], isCollapsed: true, sizePx: 360 },
        right: { activeInstanceId: 'gallery', instanceIds: ['gallery'], isCollapsed: false, sizePx: 360 },
      },
    });
    const invocation = route({ destination: 'canvas', sourceId: 'canvas' });

    expect(enter({ snapshot: structure }, undefined, invocation)).toEqual({ invocation, view: undefined });
  });
});
