import type { LayoutPresetId, WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import type { OpenIntent } from './openIntent';
import type { WorkspacePlacement } from './workspaceDerivations';

/**
 * How a widget type shows in a region, or null where it cannot. A `chip` is always on screen and opens nothing: a
 * status-bar tooltip or popover, or a center toolbar item.
 */
export type RegionPresentation = 'view' | 'chip' | null;

export interface WorkspaceCandidate {
  id: LayoutPresetId;
  /** What entering it shows for this project: its structure seen through the project's view of it. */
  arrangement: WorkspacePlacement;
  jobSourceTypeId: WidgetTypeId | null;
  /** Its stage's type while that arrangement places the stage, whatever the project last left fronted there. */
  stageTypeId: WidgetTypeId | null;
}

export interface ResolveOpenContext {
  currentWorkspaceId: LayoutPresetId;
  /** Availability and region rules for this account: disabled, hidden and admin-only types answer null everywhere. */
  getPresentation: (typeId: WidgetTypeId, region: WidgetRegion) => RegionPresentation;
  here: WorkspacePlacement;
  /** 'stay' never leaves this workspace. */
  mode: 'navigate' | 'stay';
  /** In tab order. */
  workspaces: readonly WorkspaceCandidate[];
}

export type OpenTarget =
  | { kind: 'floating'; instanceId: WidgetInstanceId }
  | { kind: 'region'; instanceId: WidgetInstanceId; region: WidgetRegion };

export type OpenDecision =
  /** On screen: focus it. A window is raised and expanded. */
  | { step: 'visible'; target: OpenTarget }
  /** Placed here behind a tab, in a closed or collapsed panel, or as a center alternate: front it. */
  | { step: 'hidden'; target: OpenTarget }
  | { step: 'elsewhere'; target: OpenTarget; workspaceId: LayoutPresetId }
  /** Nothing here serves the intent; `placedHere` when this workspace has the type on another surface. */
  | { step: 'not-placed'; placedHere: boolean }
  | { step: 'unavailable' };

interface Placement {
  isFront: boolean;
  isVisible: boolean;
  target: OpenTarget;
}

/** Rails before the status bar, the center last: a widget docked beside the work surface shows without replacing it. */
const PLACEMENT_ORDER: readonly WidgetRegion[] = ['right', 'left', 'bottom', 'center'];

const satisfiesSurface = (intent: OpenIntent, region: WidgetRegion, presentation: RegionPresentation): boolean => {
  if (presentation === 'chip') {
    return intent.surface === undefined;
  }

  return region === 'center' ? intent.surface !== 'panel' : intent.surface !== 'stage';
};

const listPlacements = (
  intent: OpenIntent,
  placement: WorkspacePlacement,
  getPresentation: ResolveOpenContext['getPresentation']
): Placement[] => {
  const isType = (instanceId: WidgetInstanceId) => placement.widgetInstances[instanceId]?.typeId === intent.typeId;
  const placements: Placement[] =
    intent.surface === 'stage'
      ? []
      : Object.keys(placement.floatingWidgets ?? {})
          .filter(isType)
          .map((instanceId) => ({ isFront: false, isVisible: true, target: { instanceId, kind: 'floating' } }));

  for (const region of PLACEMENT_ORDER) {
    const presentation = getPresentation(intent.typeId, region);

    if (presentation === null || !satisfiesSurface(intent, region, presentation)) {
      continue;
    }

    const state = placement.widgetRegions[region];
    const isShown = region === 'center' || !state.isCollapsed;

    // Every region renders its front, even one instance the center fronts too (a workflow's graph and its form).
    for (const instanceId of state.instanceIds.filter(isType)) {
      const isFront = state.activeInstanceId === instanceId;

      placements.push({
        isFront,
        isVisible: presentation === 'chip' || (isFront && isShown),
        target: { instanceId, kind: 'region', region },
      });
    }
  }

  return placements;
};

const resolvePlacedHere = (
  intent: OpenIntent,
  placement: WorkspacePlacement,
  getPresentation: ResolveOpenContext['getPresentation']
): Extract<OpenDecision, { step: 'hidden' | 'visible' }> | null => {
  const placements = listPlacements(intent, placement, getPresentation);
  const visible = placements.find((candidate) => candidate.isVisible);

  if (visible) {
    return { step: 'visible', target: visible.target };
  }

  const hidden = placements.find((candidate) => candidate.isFront) ?? placements[0];

  return hidden ? { step: 'hidden', target: hidden.target } : null;
};

const canRenderAnywhere = (intent: OpenIntent, getPresentation: ResolveOpenContext['getPresentation']): boolean => {
  if (intent.surface === 'stage') {
    return getPresentation(intent.typeId, 'center') === 'view';
  }

  return (['left', 'right', 'bottom', 'center'] as const).some((region) => {
    const presentation = getPresentation(intent.typeId, region);

    return presentation !== null && satisfiesSurface(intent, region, presentation);
  });
};

/** How central the widget is to a workspace, lower first: its stage, its job's source, a home front, placed at all. */
const rankWorkspace = (
  intent: OpenIntent,
  workspace: WorkspaceCandidate,
  getPresentation: ResolveOpenContext['getPresentation']
): number => {
  if (
    intent.surface !== 'panel' &&
    workspace.stageTypeId === intent.typeId &&
    getPresentation(intent.typeId, 'center') === 'view'
  ) {
    return 0;
  }

  if (workspace.jobSourceTypeId === intent.typeId) {
    return 1;
  }

  return listPlacements(intent, workspace.arrangement, getPresentation).some((placement) => placement.isFront) ? 2 : 3;
};

/**
 * Where an open lands. First match wins: on screen here, placed here but hidden, placed in another workspace (only in
 * 'navigate' mode), placed nowhere. A type that cannot render for this intent is unavailable. Reads placement only, so
 * built-in and custom workspaces resolve alike.
 */
export const resolveOpen = (intent: OpenIntent, context: ResolveOpenContext): OpenDecision => {
  const { getPresentation } = context;

  if (!canRenderAnywhere(intent, getPresentation)) {
    return { step: 'unavailable' };
  }

  const here = resolvePlacedHere(intent, context.here, getPresentation);

  if (here) {
    return here;
  }

  if (context.mode === 'navigate') {
    let best: { rank: number; target: OpenTarget; workspaceId: LayoutPresetId } | null = null;

    for (const workspace of context.workspaces) {
      const there =
        workspace.id === context.currentWorkspaceId
          ? null
          : resolvePlacedHere(intent, workspace.arrangement, getPresentation);
      const rank = there ? rankWorkspace(intent, workspace, getPresentation) : Infinity;

      if (there && (best === null || rank < best.rank)) {
        best = { rank, target: there.target, workspaceId: workspace.id };
      }
    }

    if (best) {
      return { step: 'elsewhere', target: best.target, workspaceId: best.workspaceId };
    }
  }

  const isType = (instanceId: WidgetInstanceId) => context.here.widgetInstances[instanceId]?.typeId === intent.typeId;
  const placedHere =
    Object.keys(context.here.floatingWidgets ?? {}).some(isType) ||
    Object.values(context.here.widgetRegions).some(({ instanceIds }) => instanceIds.some(isType));

  return { placedHere, step: 'not-placed' };
};
