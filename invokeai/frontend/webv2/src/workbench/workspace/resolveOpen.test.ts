import type { WidgetRegion } from '@workbench/layoutContracts';

import { layoutPresets } from '@workbench/layoutPresets';
import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { describe, expect, it } from 'vitest';

import type { OpenIntent } from './openIntent';
import type { ResolveOpenContext, WorkspaceCandidate } from './resolveOpen';

import { resolveOpen } from './resolveOpen';
import { createWorkspaceCandidate, getRegionPresentation } from './workspaceNavigation';

const presentation = getRegionPresentation(getWidgetsForRegion);

const builtIns: WorkspaceCandidate[] = layoutPresets.map((preset) => createWorkspaceCandidate(preset, preset.snapshot));

const resolveFrom = (
  workspaceId: string,
  intent: OpenIntent,
  {
    getPresentation = presentation,
    mode = 'navigate',
    workspaces = builtIns,
  }: Partial<Pick<ResolveOpenContext, 'getPresentation' | 'mode' | 'workspaces'>> = {}
) =>
  resolveOpen(intent, {
    currentWorkspaceId: workspaceId,
    getPresentation,
    here: workspaces.find((workspace) => workspace.id === workspaceId)!.arrangement,
    mode,
    workspaces,
  });

type RegionFixture = { front: string; members: string[]; collapsed?: boolean };

/**
 * A custom workspace whose instance ids are their widget types; panels open unless a region is collapsed. Its job is
 * derived as production derives it: the explicit route's source while placed, else the first front source.
 */
const customWorkspace = (
  id: string,
  regions: Partial<Record<WidgetRegion, RegionFixture>>,
  { floating = [], route }: { floating?: string[]; route?: 'workflow' } = {}
): WorkspaceCandidate => {
  const region = (fixture: RegionFixture | undefined) => ({
    activeInstanceId: fixture?.front ?? '',
    instanceIds: fixture?.members ?? [],
    isCollapsed: fixture?.collapsed ?? false,
  });
  const typeIds = [...Object.values(regions).flatMap((fixture) => fixture.members), ...floating];

  const arrangement = {
    floatingWidgets: Object.fromEntries(floating.map((typeId) => [typeId, {}])),
    layout: { panels: { isBottomOpen: true, isLeftOpen: true, isRightOpen: true } },
    widgetInstances: Object.fromEntries(typeIds.map((typeId) => [typeId, { typeId }])),
    widgetRegions: {
      bottom: region(regions.bottom),
      center: region(regions.center),
      left: region(regions.left),
      right: region(regions.right),
    },
  };

  return createWorkspaceCandidate(
    { defaultRoute: route ? { destination: 'gallery', sourceId: route } : undefined, id, snapshot: arrangement },
    arrangement
  );
};

const canvasOnly = customWorkspace('canvas-only', { center: { front: 'canvas', members: ['canvas'] } });

describe('resolveOpen with the built-in workspaces', () => {
  it('enters Automate to edit a workflow loaded from Compose', () => {
    expect(resolveFrom('compose', { surface: 'stage', typeId: 'workflow' })).toEqual({
      step: 'elsewhere',
      target: { instanceId: 'workflow:center', kind: 'region', region: 'center' },
      workspaceId: 'automate',
    });
  });

  it('fronts Preview in Edit’s center for Open in Preview', () => {
    expect(resolveFrom('edit', { surface: 'stage', typeId: 'preview' })).toEqual({
      step: 'hidden',
      target: { instanceId: 'preview', kind: 'region', region: 'center' },
    });
  });

  it('enters Edit to edit an image on the Canvas from Compose', () => {
    expect(resolveFrom('compose', { surface: 'stage', typeId: 'canvas' })).toMatchObject({
      step: 'elsewhere',
      workspaceId: 'edit',
    });
  });

  it('fronts Upscale in Edit’s left rail for Send to Upscale', () => {
    expect(resolveFrom('edit', { surface: 'panel', typeId: 'upscale' })).toEqual({
      step: 'hidden',
      target: { instanceId: 'upscale', kind: 'region', region: 'left' },
    });
  });

  it('enters Video, where Video is the job, to animate an image from Compose', () => {
    expect(resolveFrom('compose', { surface: 'panel', typeId: 'video' })).toEqual({
      step: 'elsewhere',
      target: { instanceId: 'video', kind: 'region', region: 'left' },
      workspaceId: 'video',
    });
  });

  it('fronts Queue in Compose’s right rail', () => {
    expect(resolveFrom('compose', { typeId: 'queue' })).toEqual({
      step: 'hidden',
      target: { instanceId: 'queue', kind: 'region', region: 'right' },
    });
  });

  it('recalls settings into Compose from Automate, and stays in Edit, which shows Generate', () => {
    expect(resolveFrom('automate', { surface: 'panel', typeId: 'generate' })).toMatchObject({
      step: 'elsewhere',
      workspaceId: 'compose',
    });
    expect(resolveFrom('edit', { surface: 'panel', typeId: 'generate' })).toEqual({
      step: 'visible',
      target: { instanceId: 'generate', kind: 'region', region: 'left' },
    });
  });

  it('prefers a rail to the center unless the stage is asked for', () => {
    expect(resolveFrom('edit', { typeId: 'preview' })).toEqual({
      step: 'hidden',
      target: { instanceId: 'preview', kind: 'region', region: 'right' },
    });
  });

  it('counts a status-bar chip as shown, but never as a panel', () => {
    expect(resolveFrom('compose', { typeId: 'server-status' })).toEqual({
      step: 'visible',
      target: { instanceId: 'server-status', kind: 'region', region: 'bottom' },
    });
    expect(resolveFrom('compose', { surface: 'panel', typeId: 'server-status' })).toEqual({ step: 'unavailable' });
  });

  it('calls a type unavailable when it is off or cannot take the asked surface', () => {
    const withoutQueue: ResolveOpenContext['getPresentation'] = (typeId, region) =>
      typeId === 'queue' ? null : presentation(typeId, region);

    expect(resolveFrom('compose', { typeId: 'queue' }, { getPresentation: withoutQueue })).toEqual({
      step: 'unavailable',
    });
    expect(resolveFrom('compose', { surface: 'stage', typeId: 'queue' })).toEqual({ step: 'unavailable' });
  });
});

describe('resolveOpen with custom workspaces', () => {
  it('ranks other workspaces by stage, job source, home front, then placement, ties in tab order', () => {
    const placedBehind = customWorkspace('placed', {
      center: { front: 'canvas', members: ['canvas'] },
      right: { front: 'gallery', members: ['gallery', 'preview'] },
    });
    const homeFront = customWorkspace('home-front', {
      center: { front: 'canvas', members: ['canvas'] },
      right: { front: 'preview', members: ['preview'] },
    });
    const secondHomeFront = customWorkspace('second-home-front', {
      center: { front: 'canvas', members: ['canvas'] },
      right: { front: 'preview', members: ['preview'] },
    });
    const stage = customWorkspace('stage', { center: { front: 'preview', members: ['preview'] } });
    const workspaceFor = (workspaces: WorkspaceCandidate[]) => {
      const decision = resolveFrom('canvas-only', { typeId: 'preview' }, { workspaces: [canvasOnly, ...workspaces] });

      return decision.step === 'elsewhere' ? decision.workspaceId : decision.step;
    };

    expect(workspaceFor([placedBehind, homeFront, secondHomeFront, stage])).toBe('stage');
    expect(workspaceFor([placedBehind, secondHomeFront, homeFront])).toBe('second-home-front');
    expect(workspaceFor([placedBehind])).toBe('placed');

    const formBehind = customWorkspace('form-behind', {
      center: { front: 'canvas', members: ['canvas'] },
      left: { front: 'generate', members: ['generate', 'workflow'] },
    });
    const job = customWorkspace(
      'job',
      {
        center: { front: 'canvas', members: ['canvas'] },
        left: { front: 'upscale', members: ['upscale', 'workflow'] },
      },
      { route: 'workflow' }
    );
    const graphInStatusBar = customWorkspace('graph-in-status-bar', {
      bottom: { front: 'workflow', members: ['workflow'] },
      center: { front: 'canvas', members: ['canvas'] },
    });

    expect(
      resolveFrom(
        'canvas-only',
        { surface: 'panel', typeId: 'workflow' },
        { workspaces: [canvasOnly, formBehind, graphInStatusBar, job] }
      )
    ).toMatchObject({ step: 'elsewhere', workspaceId: 'job' });
  });

  it('ranks a workspace by its saved stage, not by what the project last left in its center', () => {
    const edit = layoutPresets.find(({ id }) => id === 'edit')!;
    const compose = layoutPresets.find(({ id }) => id === 'compose')!;
    const editLeftOnPreview = createWorkspaceCandidate(edit, {
      ...edit.snapshot,
      widgetRegions: {
        ...edit.snapshot.widgetRegions,
        center: { ...edit.snapshot.widgetRegions.center, activeInstanceId: 'preview' },
      },
    });

    expect(
      resolveFrom(
        'canvas-only',
        { surface: 'stage', typeId: 'preview' },
        { workspaces: [canvasOnly, editLeftOnPreview, createWorkspaceCandidate(compose, compose.snapshot)] }
      )
    ).toMatchObject({ step: 'elsewhere', workspaceId: 'compose' });
  });

  it('stays where a run-as-app workspace places the workflow form', () => {
    const runAsApp = customWorkspace('run-as-app', {
      center: { front: 'preview', members: ['preview'] },
      left: { collapsed: true, front: 'workflow', members: ['workflow'] },
    });

    expect(
      resolveFrom('run-as-app', { surface: 'panel', typeId: 'workflow' }, { workspaces: [runAsApp, ...builtIns] })
    ).toEqual({ step: 'hidden', target: { instanceId: 'workflow', kind: 'region', region: 'left' } });
  });

  it('leaves the workspace only in navigate mode, and reports a widget no workspace places', () => {
    const workspaces = [canvasOnly, ...builtIns];

    expect(resolveFrom('canvas-only', { typeId: 'queue' }, { workspaces })).toMatchObject({
      step: 'elsewhere',
      workspaceId: 'automate',
    });
    expect(resolveFrom('canvas-only', { typeId: 'queue' }, { mode: 'stay', workspaces })).toEqual({
      placedHere: false,
      step: 'not-placed',
    });
    expect(resolveFrom('canvas-only', { typeId: 'diagnostics' }, { workspaces })).toEqual({
      placedHere: false,
      step: 'not-placed',
    });
  });

  it('says a widget is here on another surface when nothing here serves the asked one', () => {
    const railPreview = customWorkspace('rail-preview', {
      center: { front: 'canvas', members: ['canvas'] },
      right: { front: 'preview', members: ['preview'] },
    });

    expect(
      resolveFrom('rail-preview', { surface: 'stage', typeId: 'preview' }, { mode: 'stay', workspaces: [railPreview] })
    ).toEqual({ placedHere: true, step: 'not-placed' });
  });

  it('shows a rail copy of the center view in its rail only when a panel is asked for', () => {
    const shared = customWorkspace('shared', {
      center: { front: 'preview', members: ['preview'] },
      right: { front: 'gallery', members: ['gallery', 'preview'] },
    });
    const workspaces = [shared];

    expect(resolveFrom('shared', { typeId: 'preview' }, { workspaces })).toEqual({
      step: 'visible',
      target: { instanceId: 'preview', kind: 'region', region: 'center' },
    });
    expect(resolveFrom('shared', { surface: 'panel', typeId: 'preview' }, { workspaces })).toEqual({
      step: 'hidden',
      target: { instanceId: 'preview', kind: 'region', region: 'right' },
    });
  });

  it('finds a form on screen when its instance is also the graph the center shows', () => {
    const sharedWorkflow = customWorkspace('shared-workflow', {
      center: { front: 'workflow', members: ['preview', 'workflow'] },
      left: { front: 'workflow', members: ['generate', 'workflow'] },
    });

    expect(
      resolveFrom('shared-workflow', { surface: 'panel', typeId: 'workflow' }, { workspaces: [sharedWorkflow] })
    ).toEqual({ step: 'visible', target: { instanceId: 'workflow', kind: 'region', region: 'left' } });
  });

  it('shows a window for a panel but never for the stage', () => {
    const floatingPreview = customWorkspace(
      'floating-preview',
      { center: { front: 'canvas', members: ['canvas'] } },
      { floating: ['preview'] }
    );
    const workspaces = [floatingPreview, ...builtIns];

    expect(resolveFrom('floating-preview', { surface: 'panel', typeId: 'preview' }, { workspaces })).toEqual({
      step: 'visible',
      target: { instanceId: 'preview', kind: 'floating' },
    });
    expect(resolveFrom('floating-preview', { surface: 'stage', typeId: 'preview' }, { workspaces })).toMatchObject({
      step: 'elsewhere',
      workspaceId: 'compose',
    });
  });
});
