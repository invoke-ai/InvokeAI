import type { LayoutPresetId, WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import type { OpenIntent } from './openIntent';
import type { GuestSurface } from './view';
import type { WorkspacePlacement } from './workspaceDerivations';

import { isPlacedInstance } from './workspaceDerivations';

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
  /** The types a widget type acts on (its manifest's `needs`). */
  getNeeds: (typeId: WidgetTypeId) => readonly WidgetTypeId[];
  /** Availability and region rules for this account: disabled, hidden and admin-only types answer null everywhere. */
  getPresentation: (typeId: WidgetTypeId, region: WidgetRegion) => RegionPresentation;
  here: WorkspacePlacement;
  /** 'stay' never leaves this workspace; Customize resolves in it. */
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
  /** Placed nowhere this open may go: shown here as a guest, fronted in the center (Away) or in a window. */
  | { step: 'guest'; surface: GuestSurface }
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

/** Where the placement shows the intent; another workspace's guests are not placed there, so `guests` leaves them out. */
const listPlacements = (
  intent: OpenIntent,
  placement: WorkspacePlacement,
  getPresentation: ResolveOpenContext['getPresentation'],
  guests: 'include' | 'exclude' = 'include'
): Placement[] => {
  const isType = (instanceId: WidgetInstanceId) =>
    placement.widgetInstances[instanceId]?.typeId === intent.typeId &&
    (guests === 'include' || isPlacedInstance(placement, instanceId));
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
  getPresentation: ResolveOpenContext['getPresentation'],
  guests: 'include' | 'exclude' = 'include'
): Extract<OpenDecision, { step: 'hidden' | 'visible' }> | null => {
  const placements = listPlacements(intent, placement, getPresentation, guests);
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

  return listPlacements(intent, workspace.arrangement, getPresentation, 'exclude').some(
    (placement) => placement.isFront
  )
    ? 2
    : 3;
};

/** Whether the arrangement places a type at all, its guests aside. */
const placesType = (placement: WorkspacePlacement, typeId: WidgetTypeId): boolean => {
  const isType = (instanceId: WidgetInstanceId) =>
    placement.widgetInstances[instanceId]?.typeId === typeId && isPlacedInstance(placement, instanceId);

  return (
    Object.keys(placement.floatingWidgets ?? {}).some(isType) ||
    Object.values(placement.widgetRegions).some(({ instanceIds }) => instanceIds.some(isType))
  );
};

const isAvailable = (typeId: WidgetTypeId, getPresentation: ResolveOpenContext['getPresentation']): boolean =>
  (['left', 'right', 'bottom', 'center'] as const).some((region) => getPresentation(typeId, region) !== null);

/**
 * Step 4's surface: the center for a stage, or for a type no side region shows as a panel; a window otherwise. Null
 * for a type only a chip can show, which no surface outside the status bar can.
 */
const getGuestSurface = (
  intent: OpenIntent,
  getPresentation: ResolveOpenContext['getPresentation']
): GuestSurface | null => {
  const isCenterView = getPresentation(intent.typeId, 'center') === 'view';
  const isPanel = (['left', 'right', 'bottom'] as const).some(
    (region) => getPresentation(intent.typeId, region) === 'view'
  );

  if (intent.surface === 'stage') {
    return isCenterView ? 'center' : null;
  }

  return isPanel ? 'window' : intent.surface !== 'panel' && isCenterView ? 'center' : null;
};

/**
 * Where an open lands. First match wins: on screen here, placed here but hidden, placed in another workspace (only in
 * 'navigate' mode, and for a widget with `needs` only one that places what it needs), else a guest here. A type that
 * cannot render for this intent is unavailable. Reads placement only, so built-in and custom workspaces resolve alike.
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
    const needs = context.getNeeds(intent.typeId).filter((typeId) => isAvailable(typeId, getPresentation));
    let best: { rank: number; target: OpenTarget; workspaceId: LayoutPresetId } | null = null;

    for (const workspace of context.workspaces) {
      const there =
        workspace.id === context.currentWorkspaceId ||
        !needs.every((typeId) => placesType(workspace.arrangement, typeId))
          ? null
          : resolvePlacedHere(intent, workspace.arrangement, getPresentation, 'exclude');
      const rank = there ? rankWorkspace(intent, workspace, getPresentation) : Infinity;

      if (there && (best === null || rank < best.rank)) {
        best = { rank, target: there.target, workspaceId: workspace.id };
      }
    }

    if (best) {
      return { step: 'elsewhere', target: best.target, workspaceId: best.workspaceId };
    }
  }

  const surface = getGuestSurface(intent, getPresentation);

  return surface === null ? { step: 'unavailable' } : { step: 'guest', surface };
};
