import type { InvocationRoute } from '@workbench/invocationContracts';
import type { WidgetRegion } from '@workbench/layoutContracts';

import { getNaturalDestination } from '@workbench/graphWidgets';
import { getAvailableSourceIdForWidgetTypeId } from '@workbench/invocationSources';
import { getLayoutPreset } from '@workbench/layoutPresets';
import { isCenterViewType } from '@workbench/widgetRegistry';
import { describe, expect, it } from 'vitest';

import type { SavedWorkspace, WorkspacePlacement } from './workspaceDerivations';

import { enterWorkspace } from './entry';

const sources = { getNaturalDestination, getSourceId: getAvailableSourceIdForWidgetTypeId };
const enter = (input: Omit<Parameters<typeof enterWorkspace>[0], 'isCenterView' | 'sources'>) =>
  enterWorkspace({ ...input, isCenterView: isCenterViewType, sources });
const route = (overrides: Partial<InvocationRoute> = {}): InvocationRoute => ({
  destination: 'gallery',
  destinationLocked: false,
  sourceId: 'generate',
  sourceLocked: false,
  ...overrides,
});

/** The saved arrangement with some regions fronting another member, or collapsed. */
const laidOut = (
  workspace: SavedWorkspace,
  regions: Partial<Record<WidgetRegion, { front?: string; isCollapsed?: boolean }>>
): WorkspacePlacement => ({
  ...workspace.snapshot,
  widgetRegions: Object.fromEntries(
    Object.entries(workspace.snapshot.widgetRegions).map(([region, state]) => [
      region,
      {
        ...state,
        activeInstanceId: regions[region as WidgetRegion]?.front ?? state.activeInstanceId,
        isCollapsed: regions[region as WidgetRegion]?.isCollapsed ?? state.isCollapsed,
      },
    ])
  ) as WorkspacePlacement['widgetRegions'],
});

const edit = getLayoutPreset('edit');
const compose = getLayoutPreset('compose');
const video = getLayoutPreset('video');

describe('enterWorkspace', () => {
  it('ends Away: Edit left on Preview comes back to Canvas, aimed Canvas to Canvas', () => {
    expect(
      enter({
        invocation: route(),
        placement: laidOut(edit, { center: { front: 'preview' } }),
        workspace: edit,
      })
    ).toEqual({
      fronts: { center: 'canvas' },
      invocation: route({ destination: 'canvas', sourceId: 'canvas' }),
    });
  });

  it('keeps each locked half of the route and still restores the stage', () => {
    const entry = enter({
      invocation: route({ sourceLocked: true }),
      placement: laidOut(edit, { center: { front: 'preview' } }),
      workspace: edit,
    });

    expect(entry.fronts).toEqual({ center: 'canvas' });
    expect(entry.invocation).toEqual(route({ destination: 'canvas', sourceLocked: true }));
  });

  it('fronts the job’s source in a collapsed panel without opening it', () => {
    expect(
      enter({
        invocation: route({ sourceId: 'upscale' }),
        placement: laidOut(compose, { left: { front: 'upscale', isCollapsed: true } }),
        workspace: compose,
      })
    ).toEqual({ fronts: { left: 'generate' }, invocation: route() });
  });

  it('re-fronts Video in the Video rail after Upscale was fronted there', () => {
    expect(
      enter({
        invocation: route({ sourceId: 'upscale' }),
        placement: laidOut(video, { left: { front: 'upscale' } }),
        workspace: video,
      })
    ).toEqual({ fronts: { left: 'video' }, invocation: route({ sourceId: 'video' }) });
  });

  it('leaves the route of a workspace with no source as it was', () => {
    const review: SavedWorkspace = {
      snapshot: {
        layout: { panels: { isBottomOpen: false, isLeftOpen: false, isRightOpen: true } },
        widgetInstances: { gallery: { typeId: 'gallery' }, 'image-map': { typeId: 'image-map' } },
        widgetRegions: {
          bottom: { activeInstanceId: '', instanceIds: [], isCollapsed: true },
          center: { activeInstanceId: 'image-map', instanceIds: ['image-map'], isCollapsed: false },
          left: { activeInstanceId: '', instanceIds: [], isCollapsed: false },
          right: { activeInstanceId: 'gallery', instanceIds: ['gallery'], isCollapsed: false },
        },
      },
    };
    const invocation = route({ destination: 'canvas', sourceId: 'canvas' });

    expect(enter({ invocation, placement: review.snapshot, workspace: review })).toEqual({
      fronts: {},
      invocation,
    });
  });

  it('fronts the first center view it can show when the laid-out arrangement no longer places the stage', () => {
    const placement = laidOut(edit, {});
    const withoutCanvas: WorkspacePlacement = {
      ...placement,
      widgetRegions: {
        ...placement.widgetRegions,
        center: { activeInstanceId: 'queue', instanceIds: ['queue', 'gallery:center', 'preview'], isCollapsed: false },
      },
      widgetInstances: {
        ...placement.widgetInstances,
        'gallery:center': { typeId: 'gallery' },
        queue: { typeId: 'queue' },
      },
    };
    const invocation = route({ destination: 'canvas', sourceId: 'canvas' });

    expect(enter({ invocation, placement: withoutCanvas, workspace: edit }).fronts.center).toBe('gallery:center');
    expect(
      enterWorkspace({
        invocation,
        isCenterView: (typeId) => typeId === 'preview',
        placement: withoutCanvas,
        sources,
        workspace: edit,
      }).fronts.center
    ).toBe('preview');
  });
});
